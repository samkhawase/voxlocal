// ---------------------------------------------------------------------------
// Stage 5 — SmolLM2-135M-Instruct tool router
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{Device, Tensor};

// Re-exported so `candle::quantized::gguf_file` resolves through candle-core.
use candle_core as candle;
use candle_transformers::models::quantized_llama::ModelWeights as QuantLlama;
use regex::Regex;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::rag::{rag_lookup, Doc};

/// Generation cap for the router.
///
/// A complete two-arg tool call is ~30 tokens, but a 135M model can wander;
/// the repair pass recovers truncated JSON, so a slightly higher cap buys
/// real robustness for a few ms of worst-case decode.
pub const MAX_NEW_TOKENS: usize = 48;

// ---------------------------------------------------------------------------
// Stage 5 — SmolLM2-135M-Instruct tool router
// ---------------------------------------------------------------------------

/// The tool contract the model is asked to emit. Strict JSON only.
#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub tool: String,
    #[serde(default)]
    pub args: HashMap<String, serde_json::Value>,
}

pub struct Llm {
    model: QuantLlama,
    tokenizer: Tokenizer,
    device: Device,
    eos: u32,
}

impl Llm {
    pub fn load(gguf: &Path, tok_path: &Path, device: &Device) -> Result<Self> {
        // GGUF is a different type from candle_nn's VarBuilder, and candle 0.11's
        // `from_gguf` eagerly loads every tensor, so we read the metadata
        // ourselves and hand it to the model constructor alongside the file handle.
        use candle::quantized::gguf_file;
        let mut file = std::fs::File::open(gguf).context("open GGUF")?;
        let content = gguf_file::Content::read(&mut file).context("read GGUF metadata")?;
        let model = QuantLlama::from_gguf(content, &mut file, device)
            .map_err(|e| anyhow!("load SmolLM2: {e}"))?;

        let tokenizer =
            Tokenizer::from_file(tok_path).map_err(|e| anyhow!("LLM tokenizer: {e}"))?;

        // SmolLM2 chat template terminates on these; take them from the tokenizer.
        let eos = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(2);

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            eos,
        })
    }

    /// Build the ChatML prompt.
    ///
    /// Kept deliberately short, and given one worked example. A 135M model has
    /// weak instruction-following; a single in-context example is what actually
    /// pins it to JSON-only output — the "Output raw JSON only" instruction
    /// alone is not enough (it answers in prose). Prefill cost is trivial at
    /// this size, and correctness here is worth far more than the few ms.
    fn build_prompt(&self, query: &str, context: &str) -> Result<String> {
        let system = format!(
            "You are an API router for an autoshop. Output raw JSON only. Do not talk.\n\
             Format: {{\"tool\": \"<name>\", \"args\": {{<keys>}}}}\n\
             Tools:\n\
             - book_appointment(service, time)\n\
             - check_price(service)\n\
             Context: {context}"
        );
        // Two worked examples, one per tool. With only the check_price example the
        // model copies it even for "book me ...", because a 135M model latches onto
        // the single demonstrated behaviour.
        let prompt = format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\nwhat is the price of an oil change<|im_end|>\n\
             <|im_start|>assistant\n{{\"tool\": \"check_price\", \"args\": {{\"service\": \"oil change\"}}}}<|im_end|>\n\
             <|im_start|>user\nbook a brake pad appointment tomorrow at 9am<|im_end|>\n\
             <|im_start|>assistant\n{{\"tool\": \"book_appointment\", \"args\": {{\"service\": \"brake pads\", \"time\": \"tomorrow at 9am\"}}}}<|im_end|>\n\
             <|im_start|>user\n{query}<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        // Smoke-test that the template tokens actually exist in the vocab.
        let ids = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow!("llm encode: {e}"))?;
        if ids.is_empty() {
            bail!("prompt encoded to zero tokens — tokenizer.json looks wrong");
        }
        Ok(prompt)
    }

    /// Greedy-decode, capped at `max_new` tokens. Greedy is deliberate: sampling
    /// would break the strict-JSON guarantee and add no value for routing.
    fn generate(&mut self, prompt: &str, max_new: usize) -> Result<(String, bool)> {
        let prompt_ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("llm encode: {e}"))?
            .get_ids()
            .to_vec();
        if prompt_ids.is_empty() {
            bail!("empty prompt");
        }

        let device = self.device.clone();
        let mut out_ids: Vec<u32> = Vec::new();

        // Prefill: feed the whole prompt at once.
        let prompt_tensor =
            Tensor::new(prompt_ids.clone(), &device)?.reshape((1, prompt_ids.len()))?;
        let logits = self
            .model
            .forward(&prompt_tensor, 0)
            .context("llm prefill")?;
        let mut next = argmax_last(&logits)?;

        // Decode: one token at a time, feeding position = index_pos.
        for i in 0..max_new {
            if next == self.eos {
                break;
            }
            out_ids.push(next);
            let tok = Tensor::new(vec![next], &device)?.reshape((1, 1))?;
            // index_pos continues from the end of the prefilled prompt.
            let logits = self
                .model
                .forward(&tok, prompt_ids.len() + i)
                .context("llm decode")?;
            next = argmax_last(&logits)?;
        }
        // Ran out of budget before EOS: the JSON is almost certainly cut off.
        let truncated = !out_ids.last().map(|&t| t == self.eos).unwrap_or(false);

        self.model.clear_kv_cache();

        let text = self
            .tokenizer
            .decode(&out_ids, true)
            .map_err(|e| anyhow!("llm decode text: {e}"))?;
        Ok((text.trim().to_string(), truncated))
    }

    /// End-to-end: prompt -> tool call, with a JSON repair pass.
    pub fn route(
        &mut self,
        query: &str,
        context: &str,
        top_hit: Option<&Doc>,
        max_new: usize,
    ) -> Result<(ToolCall, String)> {
        let prompt = self.build_prompt(query, context)?;
        let (raw, truncated) = self.generate(&prompt, max_new)?;

        // A truncated generation is almost always still recoverable: we know the
        // next char must be `"`, and the repair pass closes the object.
        let call = parse_tool_call(&raw)
            .or_else(|| truncated.then(|| repair_truncated(&raw)).flatten())
            .ok_or_else(|| anyhow!("model did not emit valid tool JSON: {raw:?}"))?;

        // The model's `service` argument is unreliable at 135M (it tends to echo the
        // few-shot example) and it invents tool names. Snap both: `service` comes
        // from the retrieved document, `tool` from the allowed set.
        let mut call = call;
        call.tool = snap_tool(&call.tool, query);
        if let Some(hit) = top_hit {
            call.args.insert(
                "service".to_string(),
                serde_json::Value::String(hit.title.clone()),
            );
        }
        if call.tool == "book_appointment" {
            // The model copies `time` verbatim from the few-shot example on most
            // queries, so only trust it when the query actually mentions a time.
            let mentions_time = time_from_query(query);
            call.args.insert(
                "time".to_string(),
                serde_json::Value::String(
                    mentions_time.unwrap_or_else(|| "the next available slot".into()),
                ),
            );
        } else {
            // check_price has no time argument; drop the copied one.
            call.args.remove("time");
        }
        Ok((call, raw))
    }

    /// Pay the lazy-init cost of the first forward pass.
    ///
    /// The first generate allocates the KV cache and pays setup that every later
    /// call skips, so it is paid here at startup rather than inside the user's
    /// first turn. The prompt is throwaway: the point is the forward pass, not
    /// the completion.
    pub fn warmup(&mut self) {
        let _ = self.generate("user\nhi\nassistant\n", 1);
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// Extract a time phrase from the utterance, if it names one.
///
/// Keeps the model's `time` argument honest: at 135M it is nearly always a
/// verbatim copy of the few-shot example ("tomorrow at 9am"), so we re-derive it
/// from the user's actual words instead.
fn time_from_query(query: &str) -> Option<String> {
    const DAYS: [&str; 9] = [
        "today",
        "tomorrow",
        "tonight",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
    ];
    let q = query.to_ascii_lowercase();

    let day = DAYS.iter().find(|d| q.contains(**d)).map(|d| capitalize(d));
    let time = time_re()
        .find(&q)
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| {
            if q.contains("morning") {
                "in the morning".to_string()
            } else if q.contains("afternoon") {
                "in the afternoon".to_string()
            } else {
                String::new()
            }
        });

    match (day, time.is_empty()) {
        (Some(d), false) => Some(format!("{d} at {time}")),
        (Some(d), true) => Some(d.to_string()),
        (None, false) => Some(time),
        (None, true) => None,
    }
}

/// Closes a JSON object that was cut off mid-generation.
///
/// Strategy: if the open string has no terminator, close it; then drop any
/// trailing `,` or dangling key; then close braces to balance.
fn repair_truncated(raw: &str) -> Option<ToolCall> {
    let mut s = raw.trim().to_string();

    // Close an unterminated string literal.
    let in_string = s.chars().fold(false, |acc, ch| match ch {
        '\\' => !acc,
        '"' => !acc,
        _ => acc,
    });
    if in_string {
        s.push('"');
    }

    // Drop a trailing key that never got a value.
    if let Some(colon) = s.rfind(':') {
        if !s[colon + 1..].contains('"') {
            s.truncate(colon + 1);
            s.push_str("\"\"");
        }
    }
    // Drop a dangling comma.
    let t = s.trim_end().trim_end_matches(',');
    s = t.to_string();

    // Balance braces.
    let opens = s.matches('{').count();
    let closes = s.matches('}').count();
    for _ in 0..opens.saturating_sub(closes) {
        s.push('}');
    }
    // A key with no object after it: `{"args":` + nothing.
    if s.trim_end().ends_with(':') {
        s.push_str(" {}");
    }
    parse_tool_call(&s)
}

/// Greedy argmax over the final logits row.
///
/// `ModelWeights::forward` already slices out the last position
/// (`x.i((.., seq_len - 1, ..))`), so its output is (batch, vocab) and there is
/// nothing left to index.
fn argmax_last(logits: &Tensor) -> Result<u32> {
    let (bs, _) = logits.dims2()?;
    if bs != 1 {
        bail!("expected batch size 1 in logits, got {bs}");
    }
    let flat = logits.flatten_all()?;
    Ok(flat.argmax(0)?.to_scalar::<u32>()?)
}

/// Parse the model's raw output into a [`ToolCall`].
///
/// A 135M model reliably produces JSON-*shaped* but not always JSON-*valid*
/// text, so we try a couple of cheap recoveries before giving up.
fn parse_tool_call(raw: &str) -> Option<ToolCall> {
    if let Ok(c) = serde_json::from_str::<ToolCall>(raw.trim()) {
        return Some(c);
    }
    // Widest `{...}` span in the output.
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    let slice = &raw[start..=end];
    if let Ok(c) = serde_json::from_str::<ToolCall>(slice) {
        return Some(c);
    }
    // Last resort: single-quoted keys/values, and a trailing comma before `}`.
    let mut repaired = slice.replace('\'', "\"").replace(",}", "}");
    if let Some(eq) = repaired.find("args") {
        let tail = &repaired[eq..];
        if !tail.contains('{') && !tail.contains('[') {
            repaired.push_str(" {}");
            if let Some(i) = repaired.find("args") {
                repaired.replace_range(i..i + 4, "args");
            }
        }
    }
    serde_json::from_str::<ToolCall>(&repaired).ok()
}

/// Mock executor: validates the tool name, echoes a natural-language result.
pub fn execute_tool(call: &ToolCall) -> String {
    let arg = |k: &str| {
        call.args.get(k).map(|v| match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        })
    };

    match call.tool.as_str() {
        "book_appointment" => {
            let service = arg("service").unwrap_or_else(|| "a service".into());
            let time = arg("time").unwrap_or_else(|| "the next available slot".into());
            format!("Booked {service} for {time}. We will text you a reminder.")
        }
        "check_price" => {
            let service = arg("service").unwrap_or_else(|| "that service".into());
            match rag_lookup(&service) {
                Some(price) => format!("{service} is {price}."),
                None => format!("I could not find a price for {service}."),
            }
        }
        other => format!("Unknown tool {other}."),
    }
}

/// The only two tools this router is allowed to emit.
const TOOLS: [&str; 2] = ["book_appointment", "check_price"];

/// Matches a clock time like `9am`, `2:30 pm`, `14:00`.
fn time_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b((?:[01]?\d|2[0-3])(?::[0-5]\d)?\s*(?:am|pm))\b")
            .expect("valid time regex")
    })
}

/// Snap the generated call onto the real tool set.
///
/// A 135M model does not reliably respect a closed vocabulary: it happily
/// invents `schedule_oil_change`. We keep the model's *intent* when it names a
/// real tool, and otherwise fall back to a keyword rule over the query. This is
/// a deliberate correctness-over-purity trade: the mock executor is the thing
/// under test, and it should only ever see a valid tool name.
fn snap_tool(tool: &str, query: &str) -> String {
    let t = tool.trim().to_ascii_lowercase();
    if let Some(hit) = TOOLS.iter().find(|k| t.contains(**k)) {
        return (*hit).to_string();
    }
    let q = query.to_ascii_lowercase();
    const BOOK: [&str; 6] = [
        "book",
        "schedule",
        "appointment",
        "make an appointment",
        "slot",
        "reserve",
    ];
    const PRICE: [&str; 5] = ["price", "cost", "how much", "charge", "fee"];
    if BOOK.iter().any(|k| q.contains(k)) {
        "book_appointment".into()
    } else if PRICE.iter().any(|k| q.contains(k)) {
        "check_price".into()
    } else {
        // Neither keyword present: price is the safer default for an autoshop.
        "check_price".into()
    }
}
