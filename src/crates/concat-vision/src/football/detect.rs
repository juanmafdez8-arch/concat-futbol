// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Finding the players and the ball in a frame, and a player's silhouette.
//!
//! The detector is any Ultralytics YOLO (v8 or 11) exported to ONNX at a
//! fixed square: one input, `images`, and one output, `[1, 4 + classes,
//! anchors]`, each anchor a centre, a size and a score per class. Which
//! class is what comes from a `<model>.classes.json` beside the file - the
//! class names in order, as Ultralytics writes them - so a football model
//! (`ball`, `goalkeeper`, `player`, `referee`) and a general one (`person`
//! … `sports ball`) both work: names are mapped to [`Kind`]s, anything
//! else is ignored. Without the file the eighty COCO names are assumed.
//!
//! Two detectors may be loaded at once, the football model for the people
//! and a general model beside it for the ball, which the general models
//! find more often on broadcast footage; [`Detectors::detect`] merges them.
//!
//! A player is a few dozen pixels in a wide shot, too small for the brush
//! model to read from the whole frame. [`silhouette`] therefore crops the
//! frame round his box, has the brush model read the crop at its full
//! working size, prompts it with points down his body and on the grass
//! either side, and keeps the answer whose extent fits his box best.

use std::path::Path;
use std::sync::Mutex;

use concat_core::frame::{BYTES_PER_PIXEL, Frame};

use super::{Detection, Kind, iou};
use crate::runtime::{Input, Model};
use crate::{Brush, Mask};

/// The grey Ultralytics pads a letterboxed picture with.
const PAD: f32 = 114.0 / 255.0;
/// Below this, a person is not reported.
const PERSON_SCORE: f32 = 0.25;
/// The ball is small and often blurred: it is kept far less sure.
const BALL_SCORE: f32 = 0.08;
/// Two boxes of one kind overlapping this much are one thing.
const NMS_IOU: f32 = 0.5;

/// A frame as a detector's input tensor, with the size of the picture in
/// it: see [`Detector::prepare`].
pub struct Prepared {
    data: Vec<f32>,
    w: usize,
    h: usize,
    size: usize,
}

/// One YOLO model, loaded.
pub struct Detector {
    model: Mutex<Model>,
    size: usize,
    kinds: Vec<Option<Kind>>,
    /// Which kinds this detector is trusted for.
    wanted: fn(Kind) -> bool,
}

/// The football model, and optionally a general one for the ball.
pub struct Detectors {
    people: Detector,
    ball: Option<Detector>,
}

impl Detector {
    /// Loads a detector exported at `size` pixels square, reporting every
    /// kind it knows.
    pub fn load(file: &Path, size: usize) -> Result<Detector, String> {
        let names = class_names(file);
        Ok(Detector {
            model: Mutex::new(Model::from_file(file)?),
            size,
            kinds: names.iter().map(|name| kind_of(name)).collect(),
            wanted: |_| true,
        })
    }

    /// Whether the model knows any football kind at all.
    pub fn knows(&self, kind: Kind) -> bool {
        self.kinds.contains(&Some(kind))
    }

    /// The size to decode a source of `width` × `height` to: fitted inside
    /// the model's square by its long edge.
    pub fn input_size(&self, width: u32, height: u32) -> (u32, u32) {
        fit(self.size, width, height)
    }

    /// The picture a detector of this size reads `frame` as: fitted into
    /// the top-left of its square, the rest padded, planar and scaled to
    /// `0..=1`. Made once per frame and handed to both models, and made on
    /// the decoder's thread while the last frame is being detected.
    pub fn prepare(&self, frame: &Frame) -> Prepared {
        let size = self.size;
        let (fw, fh) = (frame.width() as usize, frame.height() as usize);
        if fw == 0 || fh == 0 {
            return Prepared { data: Vec::new(), w: 0, h: 0, size };
        }
        let (w, h) = fit(size, fw as u32, fh as u32);
        let (w, h) = (w as usize, h as usize);
        let pixels = frame.pixels();
        let mut data = vec![PAD; 3 * size * size];
        for y in 0..h {
            let sy = (y * fh / h).min(fh - 1);
            for x in 0..w {
                let sx = (x * fw / w).min(fw - 1);
                let at = (sy * fw + sx) * BYTES_PER_PIXEL;
                for c in 0..3 {
                    data[(c * size + y) * size + x] = f32::from(pixels[at + c]) / 255.0;
                }
            }
        }
        Prepared { data, w, h, size }
    }

    /// The things in `frame`, which should be [`Detector::input_size`].
    pub fn detect(&self, frame: &Frame) -> Result<Vec<Detection>, String> {
        self.detect_prepared(&self.prepare(frame))
    }

    /// The things in a picture already [`Detector::prepare`]d.
    pub fn detect_prepared(&self, prepared: &Prepared) -> Result<Vec<Detection>, String> {
        let size = self.size;
        let (w, h) = (prepared.w, prepared.h);
        if w == 0 || h == 0 {
            return Ok(Vec::new());
        }
        let data = if prepared.size == size {
            prepared.data.clone()
        } else {
            return Err("football detector: a picture prepared for another size".to_owned());
        };
        let outputs = self
            .model
            .lock()
            .map_err(|_| "football detector poisoned".to_owned())?
            .run(
                vec![Input {
                    name: "images",
                    dims: vec![1, 3, size, size],
                    data: data.into(),
                }],
                &["output0"],
            )?;
        let output = &outputs[0];
        let (rows, anchors) = match output.dims.as_slice() {
            [1, rows, anchors] => (*rows, *anchors),
            other => return Err(format!("football detector: unexpected output {other:?}")),
        };
        let classes = rows.saturating_sub(4);
        let value = |row: usize, anchor: usize| output.data[row * anchors + anchor];
        let mut found: Vec<Detection> = Vec::new();
        for anchor in 0..anchors {
            let (mut best, mut class) = (0f32, 0usize);
            for c in 0..classes {
                let score = value(4 + c, anchor);
                if score > best {
                    best = score;
                    class = c;
                }
            }
            let Some(Some(kind)) = self.kinds.get(class).copied() else {
                continue;
            };
            if !(self.wanted)(kind) {
                continue;
            }
            let floor = if kind == Kind::Ball {
                BALL_SCORE
            } else {
                PERSON_SCORE
            };
            if best < floor {
                continue;
            }
            let (cx, cy, bw, bh) = (
                value(0, anchor),
                value(1, anchor),
                value(2, anchor),
                value(3, anchor),
            );
            // Square pixels back to fractions of the picture.
            let bbox = [
                ((cx - bw * 0.5) / w as f32).clamp(0.0, 1.0),
                ((cy - bh * 0.5) / h as f32).clamp(0.0, 1.0),
                ((cx + bw * 0.5) / w as f32).clamp(0.0, 1.0),
                ((cy + bh * 0.5) / h as f32).clamp(0.0, 1.0),
            ];
            if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
                continue;
            }
            found.push(Detection {
                kind,
                bbox,
                score: best,
            });
        }
        Ok(suppress(found))
    }
}

impl Detectors {
    /// The football model at `people`, and the general model at `ball` if
    /// there is one, both exported at `size` pixels square.
    pub fn load(people: &Path, ball: Option<&Path>, size: usize) -> Result<Detectors, String> {
        let people = Detector::load(people, size)?;
        let ball = match ball {
            Some(file) => {
                let mut detector = Detector::load(file, size)?;
                detector.wanted = |kind| kind == Kind::Ball;
                detector.knows(Kind::Ball).then_some(detector)
            }
            None => None,
        };
        Ok(Detectors { people, ball })
    }

    /// A short name for what was loaded, for the analysis file.
    pub fn name(&self) -> String {
        match &self.ball {
            Some(_) => "football+general".to_owned(),
            None => "football".to_owned(),
        }
    }

    /// The size to decode a source to; see [`Detector::input_size`].
    pub fn input_size(&self, width: u32, height: u32) -> (u32, u32) {
        self.people.input_size(width, height)
    }

    /// The picture both models read `frame` as; see [`Detector::prepare`].
    pub fn prepare(&self, frame: &Frame) -> Prepared {
        self.people.prepare(frame)
    }

    /// Everything both models find in `frame`.
    pub fn detect(&self, frame: &Frame) -> Result<Vec<Detection>, String> {
        self.detect_prepared(&self.prepare(frame))
    }

    /// Everything both models find in a prepared picture.
    /// The general model is asked only when the football model has not
    /// seen the ball surely: it costs as much as the football model again,
    /// and most frames do not need it. Both read the one picture.
    pub fn detect_prepared(&self, prepared: &Prepared) -> Result<Vec<Detection>, String> {
        /// A ball the football model is this sure of needs no second look.
        const SURE_BALL: f32 = 0.3;
        let mut found = self.people.detect_prepared(prepared)?;
        let sure = found
            .iter()
            .any(|detection| detection.kind == Kind::Ball && detection.score >= SURE_BALL);
        if let Some(ball) = &self.ball
            && !sure
        {
            found.extend(ball.detect_prepared(prepared)?);
            found = suppress(found);
        }
        Ok(found)
    }

}

/// The long edge of `width` × `height` fitted to `size`.
fn fit(size: usize, width: u32, height: u32) -> (u32, u32) {
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let scale = size as f64 / w.max(h);
    (
        ((w * scale).round() as u32).clamp(1, size as u32),
        ((h * scale).round() as u32).clamp(1, size as u32),
    )
}

/// Overlapping boxes of one kind reduced to the surest, and people of
/// different kinds too, since a keeper is not also a player.
fn suppress(mut found: Vec<Detection>) -> Vec<Detection> {
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for detection in found {
        let clash = kept.iter().any(|held| {
            held.kind.is_person() == detection.kind.is_person()
                && iou(&held.bbox, &detection.bbox) > NMS_IOU
        });
        if !clash {
            kept.push(detection);
        }
    }
    kept
}

/// The class names written beside a model, or COCO's.
fn class_names(file: &Path) -> Vec<String> {
    let sidecar = file.with_extension("classes.json");
    std::fs::read(&sidecar)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Vec<String>>(&bytes).ok())
        .unwrap_or_else(|| {
            let mut coco = vec![String::new(); 80];
            coco[0] = "person".to_owned();
            coco[32] = "sports ball".to_owned();
            coco
        })
}

/// A class name as a football kind.
fn kind_of(name: &str) -> Option<Kind> {
    match name.to_ascii_lowercase().as_str() {
        "player" | "person" | "players" => Some(Kind::Player),
        "goalkeeper" | "goalie" | "keeper" => Some(Kind::Goalkeeper),
        "referee" | "ref" | "linesman" => Some(Kind::Referee),
        "ball" | "sports ball" | "football" | "soccer ball" => Some(Kind::Ball),
        _ => None,
    }
}

/// A tracked thing's silhouette over the whole source, read by the brush
/// model from a crop round its box. `frame` is the source at any size; the
/// mask returned covers the whole of it at `mask_width` pixels wide.
pub fn silhouette(
    brush: &Brush,
    frame: &Frame,
    bbox: [f32; 4],
    mask_width: u32,
) -> Result<Mask, String> {
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    let [l, t, r, b] = bbox;
    let (bw, bh) = ((r - l) * fw, (b - t) * fh);
    // A crop with the player in its middle half, never smaller than a
    // patch the model can see grass in.
    let side = (bw.max(bh) * 2.0).max(96.0);
    let (cx, cy) = ((l + r) * 0.5 * fw, (t + b) * 0.5 * fh);
    let x0 = (cx - side * 0.5).clamp(0.0, (fw - 1.0).max(0.0)) as u32;
    let y0 = (cy - side * 0.5).clamp(0.0, (fh - 1.0).max(0.0)) as u32;
    let x1 = ((cx + side * 0.5).ceil() as u32).min(frame.width());
    let y1 = ((cy + side * 0.5).ceil() as u32).min(frame.height());
    if x1 <= x0 + 1 || y1 <= y0 + 1 {
        return Err("football: the box is outside the picture".to_owned());
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let mut crop = Frame::black(cw, ch);
    {
        let source = frame.pixels();
        let target = crop.pixels_mut();
        let row = cw as usize * BYTES_PER_PIXEL;
        for y in 0..ch as usize {
            let from = ((y0 as usize + y) * frame.width() as usize + x0 as usize) * BYTES_PER_PIXEL;
            target[y * row..(y + 1) * row].copy_from_slice(&source[from..from + row]);
        }
    }
    // The box in the crop's fractions.
    let to_crop = |u: f32, v: f32| -> [f64; 2] {
        [
            f64::from(((u * fw - x0 as f32) / cw as f32).clamp(0.0, 1.0)),
            f64::from(((v * fh - y0 as f32) / ch as f32).clamp(0.0, 1.0)),
        ]
    };
    let inner = [to_crop(l, t), to_crop(r, b)];
    let (il, it) = (inner[0][0], inner[0][1]);
    let (ir, ib) = (inner[1][0], inner[1][1]);
    let mid = (il + ir) * 0.5;
    let positives = [
        [mid, it + (ib - it) * 0.25],
        [mid, it + (ib - it) * 0.45],
        [mid, it + (ib - it) * 0.7],
    ];
    let span = (ir - il).max(0.02);
    let negatives = [
        [(il - span * 0.6).max(0.0), (it + ib) * 0.5],
        [(ir + span * 0.6).min(1.0), (it + ib) * 0.5],
        [mid, (ib + (ib - it) * 0.15).min(1.0)],
    ];
    let embedding = brush.embed(&crop)?;
    let candidates = brush.prompted(&embedding, &positives, &negatives)?;
    // The answer whose extent fits the box best.
    let target_box = [il as f32, it as f32, ir as f32, ib as f32];
    let best = candidates
        .into_iter()
        .map(|(score, mask)| {
            let fit = extent(&mask)
                .map(|found| iou(&found, &target_box))
                .unwrap_or(0.0);
            (fit + score * 0.1, mask)
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, mask)| mask)
        .ok_or("football: the brush model answered nothing")?;

    // The crop's mask laid into a mask over the whole source, held to the
    // grown box so nothing beside the player comes along.
    let width = mask_width.max(16);
    let height = ((width as f32 * fh / fw).round() as u32).max(9);
    let mut out = Mask::filled(width, height, 0);
    let grown = super::grown(&bbox, 0.2);
    {
        let bytes = out.bytes_mut();
        let px0 = (grown[0].max(0.0) * width as f32) as u32;
        let px1 = ((grown[2].min(1.0) * width as f32).ceil() as u32).min(width);
        let py0 = (grown[1].max(0.0) * height as f32) as u32;
        let py1 = ((grown[3].min(1.0) * height as f32).ceil() as u32).min(height);
        for y in py0..py1 {
            let v = (y as f32 + 0.5) / height as f32;
            let cv = (v * fh - y0 as f32) / ch as f32;
            for x in px0..px1 {
                let u = (x as f32 + 0.5) / width as f32;
                let cu = (u * fw - x0 as f32) / cw as f32;
                if (0.0..=1.0).contains(&cu) && (0.0..=1.0).contains(&cv) {
                    bytes[(y * width + x) as usize] = (best.sample(cu, cv) * 255.0 + 0.5) as u8;
                }
            }
        }
    }
    Ok(out)
}

/// Where a shot's crop came from and went: its source pixel corner, the
/// scale it was drawn at, and its tile's corner on the canvas.
type Placement = (u32, u32, f32, u32, u32);

/// Silhouettes for up to four shots at a time - a frame and the box of the
/// thing in it - read by the brush model in one pass: the crops round the
/// boxes are laid out two by two on one 1024 square, the encoder - the
/// expensive half - reads the square once, and the decoder is prompted
/// tile by tile. Each answer is a quarter of the resolution a crop of its
/// own would get, which a feathered mark does not show, at a quarter of
/// the cost. `None` where the model found nothing that fits its box.
pub fn silhouettes(
    brush: &Brush,
    shots: &[(&Frame, [f32; 4])],
    mask_width: u32,
) -> Result<Vec<Option<Mask>>, String> {
    const CANVAS: u32 = 1024;
    const TILE: u32 = CANVAS / 2;
    let mut out = Vec::with_capacity(shots.len());
    for group in shots.chunks(4) {
        let mut canvas = Frame::black(CANVAS, CANVAS);
        // Where each shot's crop came from and where it went: the source
        // pixel box, the scale, and the tile's corner on the canvas.
        let mut placed: Vec<Option<Placement>> = Vec::new();
        for (index, (frame, bbox)) in group.iter().enumerate() {
            let (fw, fh) = (frame.width() as f32, frame.height() as f32);
            let [l, t, r, b] = *bbox;
            let side = (((r - l) * fw).max((b - t) * fh) * 2.0).max(96.0);
            let (cx, cy) = ((l + r) * 0.5 * fw, (t + b) * 0.5 * fh);
            let x0 = (cx - side * 0.5).clamp(0.0, (fw - 1.0).max(0.0)) as u32;
            let y0 = (cy - side * 0.5).clamp(0.0, (fh - 1.0).max(0.0)) as u32;
            let x1 = ((cx + side * 0.5).ceil() as u32).min(frame.width());
            let y1 = ((cy + side * 0.5).ceil() as u32).min(frame.height());
            if x1 <= x0 + 1 || y1 <= y0 + 1 {
                placed.push(None);
                continue;
            }
            let scale = TILE as f32 / (x1 - x0).max(y1 - y0) as f32;
            let (ox, oy) = ((index as u32 % 2) * TILE, (index as u32 / 2) * TILE);
            let (sw, sh) = (
                ((x1 - x0) as f32 * scale) as u32,
                ((y1 - y0) as f32 * scale) as u32,
            );
            let source = frame.pixels();
            let width = frame.width() as usize;
            let target = canvas.pixels_mut();
            for y in 0..sh.min(TILE) {
                let sy = (y0 + (y as f32 / scale) as u32).min(y1 - 1) as usize;
                for x in 0..sw.min(TILE) {
                    let sx = (x0 + (x as f32 / scale) as u32).min(x1 - 1) as usize;
                    let from = (sy * width + sx) * BYTES_PER_PIXEL;
                    let to = (((oy + y) * CANVAS + ox + x) as usize) * BYTES_PER_PIXEL;
                    target[to..to + BYTES_PER_PIXEL]
                        .copy_from_slice(&source[from..from + BYTES_PER_PIXEL]);
                }
            }
            placed.push(Some((x0, y0, scale, ox, oy)));
        }
        let embedding = brush.embed(&canvas)?;
        for ((frame, bbox), place) in group.iter().zip(&placed) {
            let Some((x0, y0, scale, ox, oy)) = *place else {
                out.push(None);
                continue;
            };
            let (fw, fh) = (frame.width() as f32, frame.height() as f32);
            // A source fraction to a canvas fraction.
            let to_canvas = |u: f32, v: f32| -> (f32, f32) {
                (
                    ((u * fw - x0 as f32) * scale + ox as f32) / CANVAS as f32,
                    ((v * fh - y0 as f32) * scale + oy as f32) / CANVAS as f32,
                )
            };
            let [l, t, r, b] = *bbox;
            let (il, it) = to_canvas(l, t);
            let (ir, ib) = to_canvas(r, b);
            let (tile_l, tile_t) = (ox as f32 / CANVAS as f32, oy as f32 / CANVAS as f32);
            let (tile_r, tile_b) = (tile_l + 0.5, tile_t + 0.5);
            let clamp_x = |x: f32| f64::from(x.clamp(tile_l, tile_r - 1e-3));
            let clamp_y = |y: f32| f64::from(y.clamp(tile_t, tile_b - 1e-3));
            let mid = (il + ir) * 0.5;
            let span = (ir - il).max(0.01);
            let positives = [
                [clamp_x(mid), clamp_y(it + (ib - it) * 0.25)],
                [clamp_x(mid), clamp_y(it + (ib - it) * 0.45)],
                [clamp_x(mid), clamp_y(it + (ib - it) * 0.7)],
            ];
            let negatives = [
                [clamp_x(il - span * 0.6), clamp_y((it + ib) * 0.5)],
                [clamp_x(ir + span * 0.6), clamp_y((it + ib) * 0.5)],
                [clamp_x(mid), clamp_y(ib + (ib - it) * 0.15)],
            ];
            let candidates = brush.prompted(&embedding, &positives, &negatives)?;
            let target_box = [il, it, ir, ib];
            let best = candidates
                .into_iter()
                .map(|(score, mask)| {
                    let fit = extent(&mask)
                        .map(|found| iou(&found, &target_box))
                        .unwrap_or(0.0);
                    (fit, score, mask)
                })
                .max_by(|a, b| (a.0 + a.1 * 0.1).total_cmp(&(b.0 + b.1 * 0.1)));
            let Some((_, _, best)) = best.filter(|(fit, _, _)| *fit > 0.2) else {
                out.push(None);
                continue;
            };
            // Laid into a mask over the whole source, held to the grown box
            // so a neighbour in the same tile never comes along.
            let width = mask_width.max(16);
            let height = ((width as f32 * fh / fw).round() as u32).max(9);
            let mut mask = Mask::filled(width, height, 0);
            let grown = super::grown(bbox, 0.2);
            let bytes = mask.bytes_mut();
            let px0 = (grown[0].max(0.0) * width as f32) as u32;
            let px1 = ((grown[2].min(1.0) * width as f32).ceil() as u32).min(width);
            let py0 = (grown[1].max(0.0) * height as f32) as u32;
            let py1 = ((grown[3].min(1.0) * height as f32).ceil() as u32).min(height);
            for y in py0..py1 {
                let v = (y as f32 + 0.5) / height as f32;
                for x in px0..px1 {
                    let u = (x as f32 + 0.5) / width as f32;
                    let (cu, cv) = to_canvas(u, v);
                    if (tile_l..tile_r).contains(&cu) && (tile_t..tile_b).contains(&cv) {
                        bytes[(y * width + x) as usize] = (best.sample(cu, cv) * 255.0 + 0.5) as u8;
                    }
                }
            }
            out.push(Some(mask));
        }
    }
    Ok(out)
}

/// The fractions a mask's sure half spans, `[left, top, right, bottom]`.
fn extent(mask: &Mask) -> Option<[f32; 4]> {
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let bytes = mask.bytes();
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if bytes[y * w + x] >= 128 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    (x0 != usize::MAX).then(|| {
        [
            x0 as f32 / w as f32,
            y0 as f32 / h as f32,
            (x1 + 1) as f32 / w as f32,
            (y1 + 1) as f32 / h as f32,
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_map_to_kinds() {
        assert_eq!(kind_of("sports ball"), Some(Kind::Ball));
        assert_eq!(kind_of("Goalkeeper"), Some(Kind::Goalkeeper));
        assert_eq!(kind_of("car"), None);
    }

    #[test]
    fn a_wide_frame_fits_by_its_width() {
        assert_eq!(fit(1280, 1920, 1080), (1280, 720));
    }

    #[test]
    fn overlapping_boxes_are_one() {
        let a = Detection {
            kind: Kind::Player,
            bbox: [0.1, 0.1, 0.2, 0.3],
            score: 0.9,
        };
        let b = Detection { score: 0.5, ..a };
        let ball = Detection {
            kind: Kind::Ball,
            ..a
        };
        assert_eq!(suppress(vec![a, b, ball]).len(), 2);
    }
}
