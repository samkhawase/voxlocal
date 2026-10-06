// ---------------------------------------------------------------------------
// Stage 4 — in-memory RAG (all-MiniLM-L6-v2 + exact cosine search)
// ---------------------------------------------------------------------------

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use tokenizers::Tokenizer;

pub struct Doc {
    /// Index into the source corpus, kept for logging/debugging.
    #[allow(dead_code)]
    id: usize,
    pub title: String,
    pub body: String,
    embedding: Vec<f32>,
}

/// MiniLM encoder producing sentence-transformers-compatible embeddings:
/// masked mean pooling over the last hidden state, then L2 normalisation.
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let cfg_bytes =
            std::fs::read(dir.join("config.json")).context("read MiniLM config.json")?;
        // Deserialize the *real* config: Config::default() is BERT-base (768/12),
        // not MiniLM (384/6).
        let cfg: BertConfig = serde_json::from_slice(&cfg_bytes).context("parse MiniLM config")?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                device,
            )
        }
        .context("mmap MiniLM weights")?;

        let model = BertModel::load(vb, &cfg).context("load BertModel")?;

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("MiniLM tokenizer: {e}"))?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    /// Embed a batch of strings. Returns L2-normalised vectors.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // Pad to a single length so we can build one dense batch tensor.
        let encodings: Vec<_> = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("MiniLM encode_batch: {e}"))?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        let (bs, seq) = (encodings.len(), max_len);

        let mut ids = vec![0u32; bs * seq];
        let mut mask = vec![0u32; bs * seq];
        for (i, e) in encodings.iter().enumerate() {
            let e_len = e.get_ids().len();
            ids[i * seq..i * seq + e_len].copy_from_slice(e.get_ids());
            mask[i * seq..i * seq + e_len].copy_from_slice(e.get_attention_mask());
        }

        let input_ids = Tensor::new(ids, &self.device)?.reshape((bs, seq))?;
        let attention = Tensor::new(mask, &self.device)?.reshape((bs, seq))?;
        // single-segment classification model: token_type_ids are all zero.
        let token_types = Tensor::zeros((bs, seq), DType::U32, &self.device)?;

        // candle's BertModel::forward takes (input_ids, token_type_ids, attention_mask).
        let hidden = self
            .model
            .forward(&input_ids, &token_types, Some(&attention))
            .context("MiniLM forward")?; // (bs, seq, hidden)

        // Masked mean pooling: sum over tokens / count of real tokens.
        let m = attention.to_dtype(DType::F32)?.unsqueeze(2)?; // (bs, seq, 1)
                                                               // broadcast_mul (not mul): lhs is (bs, seq, hidden), rhs is (bs, seq, 1).
        let summed = hidden.broadcast_mul(&m)?.sum(1)?; // (bs, hidden)
        let counts = m.sum(1)?; // (bs, 1)
        let pooled = summed.broadcast_div(&counts)?.contiguous()?;

        // L2-normalise so a dot product *is* cosine similarity.
        let dims = pooled.dims2()?.1;
        let flat = pooled
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut out = Vec::with_capacity(bs);
        for row in flat.chunks_exact(dims) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            let inv = if norm > 1e-9 { 1.0 / norm } else { 0.0 };
            out.push(row.iter().map(|v| v * inv).collect());
        }
        Ok(out)
    }

    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed_batch(&[text])?.remove(0))
    }
}

/// Minimum cosine similarity for a document to be considered relevant.
///
/// Brute-force search always returns *something*, so without a floor an
/// unrelated utterance ("hi") still gets near-zero-scoring documents spliced into
/// the prompt, which actively misleads the router. Tuned against this corpus:
/// genuine questions score >=0.55, chit-chat lands around 0.10-0.15.
pub const RAG_MIN_SIM: f32 = 0.35;

/// The autoshop knowledge base, embedded once at startup.
pub struct Rag {
    docs: Vec<Doc>,
}

impl Rag {
    pub fn build(embedder: &Embedder, corpus: &[(&str, &str)]) -> Result<Self> {
        let titles: Vec<&str> = corpus.iter().map(|(t, _)| *t).collect();
        let bodies: Vec<&str> = corpus.iter().map(|(_, b)| *b).collect();

        // Embed the whole corpus in one batch.
        let mut docs = Vec::with_capacity(corpus.len());
        let mut title_vecs = embedder.embed_batch(&titles)?;
        let body_vecs = embedder.embed_batch(&bodies)?;

        // A document's vector is the normalised mean of its title and body vectors.
        for (i, (title, body)) in corpus.iter().enumerate() {
            let mut v: Vec<f32> = title_vecs[i]
                .iter()
                .zip(body_vecs[i].iter())
                .map(|(a, b)| a + b)
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-9 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            docs.push(Doc {
                id: i,
                title: (*title).to_string(),
                body: (*body).to_string(),
                embedding: v,
            });
        }
        title_vecs.clear();
        Ok(Self { docs })
    }

    /// Exact brute-force cosine search (both vectors are unit-norm, so dot == cosine).
    ///
    /// Drops anything scoring below [`RAG_MIN_SIM`]; returns an empty list when
    /// nothing is relevant, which the caller treats as "no context".
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(&Doc, f32)> {
        let mut scored: Vec<(&Doc, f32)> = self
            .docs
            .iter()
            .map(|d| {
                let sim: f32 = d
                    .embedding
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                (d, sim)
            })
            .filter(|(_, sim)| *sim >= RAG_MIN_SIM)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

/// The autoshop corpus. Small and domain-specific on purpose: it is what the
/// router is allowed to know about.
pub fn autoshop_corpus() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "oil change",
            "Full synthetic oil change is $79 including filter. Takes about 45 minutes.",
        ),
        (
            "tire rotation",
            "Tire rotation is $35 for four tires. Recommended every 6,000 miles.",
        ),
        (
            "brake pads",
            "Front brake pads are $189 installed per axle. Ceramic or semi-metallic available.",
        ),
        (
            "battery replacement",
            "AGM battery replacement is $249 installed, includes 3 year warranty and old battery return.",
        ),
        (
            "wheel alignment",
            "Four wheel alignment is $99. Recommended annually or after suspension work.",
        ),
        (
            "air conditioning service",
            "A/C recharge and leak check is $149. Freon included up to one pound.",
        ),
        (
            "state inspection",
            "Emissions state inspection is $39 for cars and $49 for trucks and SUVs.",
        ),
        (
            "diagnostic fee",
            "Diagnostic fee is $89 and is waived if you approve the recommended repair at our shop.",
        ),
        (
            "appointment booking",
            "Appointments can be booked Monday to Saturday, 8am to 6pm. Same day slots open at 9am.",
        ),
        (
            "warranty",
            "All labor carries a 24 month parts and labor warranty. Towing included for 12 months.",
        ),
    ]
}

/// Pull a dollar figure out of the corpus for a rough spoken price answer.
pub fn rag_lookup(service: &str) -> Option<String> {
    let want = service.to_lowercase();
    autoshop_corpus()
        .iter()
        .find(|(title, _)| {
            let t = title.to_lowercase();
            want.contains(&t) || t.contains(&want)
        })
        .and_then(|(_, body)| {
            // First "$<digits>" in the body.
            let idx = body.find('$')?;
            let rest = &body[idx + 1..];
            let end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            Some(format!("${}", &rest[..end]))
        })
}
