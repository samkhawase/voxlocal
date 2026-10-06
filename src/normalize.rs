// ---------------------------------------------------------------------------
// Stage 3 — text normalisation (regex)
// ---------------------------------------------------------------------------

use anyhow::Result;
use regex::Regex;

/// A frozen, pre-compiled set of normalisation rules.
///
/// Regexes are compiled once at startup: `regex` compiles are expensive enough
/// that doing them per-utterance would show up in the latency budget.
pub struct Normalizer {
    /// Filler words, as whole words, case-insensitive.
    fillers: Regex,
    /// Thousands separators: `1,200` -> `1200`.
    thousands: Regex,
    /// Ordinals: `3rd` -> `3`.
    ordinal: Regex,
    /// Simple `YYYY-MM-DD` and `MM/DD/YYYY` dates -> spoken form.
    iso_date: Regex,
    us_date: Regex,
    /// Collapse runs of whitespace.
    space: Regex,
}

impl Normalizer {
    pub fn new() -> Result<Self> {
        let word = r"(?i)\b(um+|uh+|erm+|ah+|hmm+|mhm+|uh-huh|you know|i mean)\b[,]?";
        let y = r"(\d{4})-(\d{2})-(\d{2})";
        let us = r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b";
        Ok(Self {
            fillers: Regex::new(word)?,
            thousands: Regex::new(r"\b(\d{1,3}),(\d{3})\b")?,
            ordinal: Regex::new(r"(?i)\b(\d+)(st|nd|rd|th)\b")?,
            iso_date: Regex::new(y)?,
            us_date: Regex::new(us)?,
            space: Regex::new(r"\s+")?,
        })
    }

    pub fn apply(&self, text: &str) -> String {
        let mut s = text.to_string();

        // Dates first: their digit groups must not be eaten by the number rules.
        s = self
            .iso_date
            .replace_all(&s, |c: &regex::Captures| {
                let (y, m, d) = (
                    &c[1],
                    c[2].parse::<u32>().unwrap_or(1),
                    c[3].parse::<u32>().unwrap_or(1),
                );
                format!("{} {} {}", month_name(m), ordinal_day(d), y)
            })
            .into_owned();
        s = self
            .us_date
            .replace_all(&s, |c: &regex::Captures| {
                let (m, d, y) = (
                    c[1].parse::<u32>().unwrap_or(1),
                    c[2].parse::<u32>().unwrap_or(1),
                    &c[3],
                );
                format!("{} {} {}", month_name(m), ordinal_day(d), y)
            })
            .into_owned();

        // Then number shapes, then fillers, then whitespace.
        s = self.thousands.replace_all(&s, "$1$2").into_owned();
        s = self.ordinal.replace_all(&s, "$1").into_owned();
        s = self.fillers.replace_all(&s, " ").into_owned();
        s = self.space.replace_all(&s, " ").into_owned();

        let mut out = s.trim().to_string();
        // Collapse " ," and " ." left behind by filler removal.
        out = out.replace(" ,", ",").replace(" .", ".");
        out
    }
}

fn month_name(m: u32) -> &'static str {
    const NAMES: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    NAMES[(m.clamp(1, 12) - 1) as usize]
}

fn ordinal_day(d: u32) -> String {
    let s = d.to_string();
    let last = d % 100;
    let suffix = match (last % 10, last % 100) {
        (1, 11) => "",
        (2, 12) => "",
        (3, 13) => "",
        (n, _) if (11..=13).contains(&n) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    format!("{s}{suffix}")
}
