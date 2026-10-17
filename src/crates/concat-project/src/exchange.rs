// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! A timeline's plays in the formats other analysis tools read and write.
//!
//! **Sportscode XML** is what Hudl Sportscode calls an "XML edit list" and
//! what Nacsport, LongoMatch and most coding tools import and export: a
//! `<file>` holding `<ALL_INSTANCES>`, one `<instance>` per register with
//! its `<ID>`, `<start>` and `<end>` in seconds and its category as
//! `<code>`, each qualifier as a `<label>` with a `<group>` and a `<text>`,
//! and `<ROWS>` giving each code its colour as sixteen-bit RGB. A play's
//! descriptors go out as labels in the "Descriptor" group and its rating
//! in the "Rating" group; reading, any label that is not a rating becomes
//! a descriptor, whatever its group.
//!
//! The format is loose - Sportscode itself does not escape everything -
//! so the reader is too: it finds the elements by name, ignores what it
//! does not know, and skips an instance without a usable time rather than
//! failing the file.
//!
//! **CSV** is for a spreadsheet: one row per play, comma separated, with
//! a header, quoted where a field needs it.

use crate::model::{Play, TagCategory, TagTemplate};

/// The label group a play's descriptors are written under.
const DESCRIPTOR_GROUP: &str = "Descriptor";
/// The label group a play's rating is written under.
const RATING_GROUP: &str = "Rating";

/// The plays as a Sportscode XML edit list, coloured by `template`.
pub fn to_sportscode(plays: &[Play], template: &TagTemplate) -> String {
    let mut sorted: Vec<&Play> = plays.iter().collect();
    sorted.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<file>\n");
    out.push_str("  <SESSION_INFO>\n    <start_time>0</start_time>\n  </SESSION_INFO>\n");
    out.push_str("  <ALL_INSTANCES>\n");
    for (index, play) in sorted.iter().enumerate() {
        out.push_str("    <instance>\n");
        out.push_str(&format!("      <ID>{}</ID>\n", index + 1));
        out.push_str(&format!("      <start>{:.3}</start>\n", play.start));
        out.push_str(&format!("      <end>{:.3}</end>\n", play.end));
        out.push_str(&format!("      <code>{}</code>\n", escape(&code_of(play))));
        for descriptor in &play.descriptors {
            out.push_str(&label(DESCRIPTOR_GROUP, descriptor));
        }
        if play.rating > 0 {
            out.push_str(&label(RATING_GROUP, &play.rating.to_string()));
        }
        out.push_str("    </instance>\n");
    }
    out.push_str("  </ALL_INSTANCES>\n  <ROWS>\n");
    let mut codes: Vec<String> = Vec::new();
    for play in &sorted {
        let code = code_of(play);
        if !codes.contains(&code) {
            codes.push(code);
        }
    }
    for code in codes {
        let [r, g, b] = template
            .categories
            .iter()
            .find(|category| category.name == code)
            .map_or([0x95, 0xa5, 0xa6], |category| category.colour);
        // Sixteen bits a channel, as Sportscode writes them.
        let wide = |channel: u8| u32::from(channel) * 257;
        out.push_str(&format!(
            "    <row>\n      <code>{}</code>\n      <R>{}</R>\n      <G>{}</G>\n      <B>{}</B>\n    </row>\n",
            escape(&code),
            wide(r),
            wide(g),
            wide(b)
        ));
    }
    out.push_str("  </ROWS>\n</file>\n");
    out
}

/// What a Sportscode XML edit list holds, read back: its plays, and a
/// category for each code with the colour its row gives it.
#[derive(Clone, Debug, PartialEq)]
pub struct Imported {
    /// The plays, ids minted from `p1` upward after `first_id`.
    pub plays: Vec<Play>,
    /// One category per code, in order of first appearance.
    pub categories: Vec<TagCategory>,
}

/// Reads a Sportscode XML edit list. Plays are numbered from `first_id`,
/// so they can join a timeline's own without clashing. An error only when
/// the text holds no instance at all.
pub fn from_sportscode(text: &str, first_id: u32) -> Result<Imported, String> {
    let colours: Vec<(String, [u8; 3])> = elements(text, "row")
        .into_iter()
        .filter_map(|row| {
            let code = unescape(first(row, "code")?.trim());
            let narrow = |name: &str| -> u8 {
                let value: u32 = first(row, name)
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                // Sixteen-bit or eight-bit, whichever the writer used.
                if value > 255 { (value / 257).min(255) as u8 } else { value as u8 }
            };
            Some((code, [narrow("R"), narrow("G"), narrow("B")]))
        })
        .collect();
    let instances = elements(text, "instance");
    if instances.is_empty() {
        return Err("no <instance> in the file: not a Sportscode XML edit list".to_owned());
    }
    let mut plays: Vec<Play> = Vec::new();
    let mut categories: Vec<TagCategory> = Vec::new();
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for instance in instances {
        let time = |name: &str| first(instance, name).and_then(|v| v.trim().parse::<f64>().ok());
        let (Some(start), Some(end)) = (time("start"), time("end")) else {
            continue;
        };
        let code = first(instance, "code")
            .map(|code| unescape(code.trim()))
            .filter(|code| !code.is_empty())
            .unwrap_or_else(|| "Jugada".to_owned());
        let mut descriptors = Vec::new();
        let mut rating = 0u8;
        for found in elements(instance, "label") {
            let group = first(found, "group").map(|g| unescape(g.trim())).unwrap_or_default();
            let Some(value) = first(found, "text").map(|t| unescape(t.trim())) else {
                continue;
            };
            if group.eq_ignore_ascii_case(RATING_GROUP) {
                rating = value.parse::<u8>().unwrap_or(0).min(5);
            } else if !value.is_empty() {
                descriptors.push(value);
            }
        }
        if !categories.iter().any(|category| category.name == code) {
            let colour = colours
                .iter()
                .find(|(name, _)| *name == code)
                .map_or([0x95, 0xa5, 0xa6], |(_, colour)| *colour);
            categories.push(TagCategory::new(&code, colour));
        }
        let count = counts.entry(code.clone()).or_insert(0);
        *count += 1;
        let Some(play) = (Play {
            id: format!("p{}", first_id + plays.len() as u32),
            name: format!("{code} {count}"),
            tag: code,
            start,
            end,
            descriptors,
            rating,
        })
        .tidy() else {
            continue;
        };
        plays.push(play);
    }
    Ok(Imported { plays, categories })
}

/// The plays as CSV for a spreadsheet: start, end and length in seconds
/// and as a match clock, category, name, descriptors and rating.
pub fn to_csv(plays: &[Play]) -> String {
    let mut sorted: Vec<&Play> = plays.iter().collect();
    sorted.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut out = String::from(
        "start_s,end_s,length_s,start,category,name,descriptors,rating\n",
    );
    for play in sorted {
        let clock = {
            let whole = play.start.max(0.0).round() as u64;
            format!("{}:{:02}", whole / 60, whole % 60)
        };
        let fields = [
            format!("{:.2}", play.start),
            format!("{:.2}", play.end),
            format!("{:.2}", play.end - play.start),
            clock,
            play.tag.clone(),
            play.name.clone(),
            play.descriptors.join(" | "),
            if play.rating > 0 { play.rating.to_string() } else { String::new() },
        ];
        let row: Vec<String> = fields.iter().map(|field| csv_field(field)).collect();
        out.push_str(&row.join(","));
        out.push('\n');
    }
    out
}

fn code_of(play: &Play) -> String {
    if play.tag.is_empty() {
        play.name.clone()
    } else {
        play.tag.clone()
    }
}

fn label(group: &str, text: &str) -> String {
    format!(
        "      <label>\n        <group>{}</group>\n        <text>{}</text>\n      </label>\n",
        escape(group),
        escape(text)
    )
}

fn csv_field(field: &str) -> String {
    if field.contains([',', '"', '\n']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_owned()
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// The inner text of every `<name>…</name>` in `text`, outermost first,
/// not looking inside one for another of the same name.
fn elements<'a>(text: &'a str, name: &str) -> Vec<&'a str> {
    let open = format!("<{name}>");
    let open_attrs = format!("<{name} ");
    let close = format!("</{name}>");
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let at = match (rest.find(&open), rest.find(&open_attrs)) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => break,
        };
        let Some(gt) = rest[at..].find('>') else {
            break;
        };
        let body_start = at + gt + 1;
        let Some(end) = rest[body_start..].find(&close) else {
            break;
        };
        out.push(&rest[body_start..body_start + end]);
        rest = &rest[body_start + end + close.len()..];
    }
    out
}

/// The inner text of the first `<name>…</name>` in `text`.
fn first<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    elements(text, name).into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(id: &str, tag: &str, start: f64, end: f64) -> Play {
        Play {
            id: id.to_owned(),
            name: format!("{tag} 1"),
            tag: tag.to_owned(),
            start,
            end,
            descriptors: vec!["Carril izquierdo".to_owned(), "Éxito".to_owned()],
            rating: 4,
        }
    }

    #[test]
    fn plays_survive_a_round_trip_through_sportscode_xml() {
        let template = TagTemplate::football();
        let plays = vec![
            play("p2", "Gol", 30.0, 38.5),
            play("p1", "Presión & robo", 10.0, 14.0),
        ];
        let xml = to_sportscode(&plays, &template);
        assert!(xml.contains("<code>Presión &amp; robo</code>"));
        let back = from_sportscode(&xml, 1).expect("reads");
        assert_eq!(back.plays.len(), 2);
        assert_eq!(back.plays[0].tag, "Presión & robo");
        assert_eq!((back.plays[0].start, back.plays[0].end), (10.0, 14.0));
        assert_eq!(back.plays[1].rating, 4);
        assert_eq!(back.plays[1].descriptors.len(), 2);
        let gol = back.categories.iter().find(|c| c.name == "Gol").expect("a category");
        assert_eq!(gol.colour, [0xcb, 0xf5, 0x3f]);
    }

    #[test]
    fn a_sportscode_file_with_its_own_habits_reads() {
        // As Sportscode and Nacsport write it: no declaration, attributes on
        // the root, free text, labels in groups of their own, eight-bit
        // colours, and an instance with no end, which is skipped.
        let xml = r#"<file version="2"><ALL_INSTANCES>
            <instance><ID>7</ID><start>1.5</start><end>9</end><code>Corner</code>
              <label><group>Side</group><text>Left</text></label>
              <label><text>Short</text></label>
            </instance>
            <instance><ID>8</ID><start>20</start><code>Corner</code></instance>
          </ALL_INSTANCES>
          <ROWS><row><code>Corner</code><R>255</R><G>0</G><B>0</B></row></ROWS></file>"#;
        let read = from_sportscode(xml, 5).expect("reads");
        assert_eq!(read.plays.len(), 1);
        assert_eq!(read.plays[0].id, "p5");
        assert_eq!(read.plays[0].descriptors, vec!["Left".to_owned(), "Short".to_owned()]);
        assert_eq!(read.categories[0].colour, [255, 0, 0]);
    }

    #[test]
    fn a_file_with_no_instances_is_refused() {
        assert!(from_sportscode("<html>nothing</html>", 1).is_err());
    }

    #[test]
    fn csv_quotes_what_needs_quoting() {
        let mut odd = play("p1", "Ataque, organizado", 65.0, 70.0);
        odd.name = "Dijo \"gol\"".to_owned();
        let csv = to_csv(&[odd]);
        let row = csv.lines().nth(1).expect("a row");
        assert!(row.starts_with("65.00,70.00,5.00,1:05,\"Ataque, organizado\",\"Dijo \"\"gol\"\"\""));
        assert!(row.ends_with(",4"));
    }
}
