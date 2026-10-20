// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Tactical drawings rasterised onto a frame: the lines, arrows, curved
//! runs, zones and player links an analyst telestrates over a play.
//!
//! Everything arrives in source fractions, like every mark here, and is
//! taken to the frame through the clip's [`Mapping`] - but without the
//! clip the crop leaves, since a line may well run off the picture.
//! A path's pixels are gathered as coverage in one buffer over the path's
//! box and blended once, so a joint or a dash's overlap is never painted
//! twice; edges are a pixel soft.
//!
//! A drawing can draw itself on: `reveal` is how much of its length shows,
//! the arrowhead riding the tip as it grows, and a zone fading in.

use concat_core::frame::{BYTES_PER_PIXEL, Frame};

use crate::Mapping;

/// What is drawn through a drawing's points.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// A straight line.
    Line,
    /// A straight line with an arrowhead.
    Arrow,
    /// A bowed line with an arrowhead.
    Curve,
    /// A closed area, faintly filled.
    Zone,
    /// A line through every point, with a dot on each.
    Link,
}

/// One drawing, ready to rasterise.
#[derive(Clone, Debug)]
pub struct Sketch {
    /// What it is.
    pub kind: Kind,
    /// Its points, in source fractions.
    pub points: Vec<(f32, f32)>,
    /// Dashed rather than solid.
    pub dashed: bool,
    /// Its colour.
    pub colour: [u8; 3],
    /// Its weight against the default.
    pub thickness: f32,
    /// How far a curve bows, as a fraction of its length.
    pub bend: f32,
    /// How much of it shows, `0..=1`: the draw-on.
    pub reveal: f32,
    /// How solid it is, `0..=1`: the fade at its end.
    pub opacity: f32,
}

/// The source fraction to frame pixels, unclipped: the inverse of
/// [`Mapping::source_of`] without the refusal outside the crop.
fn to_pixels(mapping: &Mapping, frame: &Frame, (u, v): (f32, f32)) -> (f32, f32) {
    let [left, top, right, bottom] = mapping.crop;
    let x = (u - left) / (1.0 - left - right).max(1e-6);
    let y = (v - top) / (1.0 - top - bottom).max(1e-6);
    let x = if mapping.flip_h { 1.0 - x } else { x };
    let y = if mapping.flip_v { 1.0 - y } else { y };
    (x * frame.width() as f32, y * frame.height() as f32)
}

/// A quadratic bow from `a` to `b`, bent `bend` of its length to the left
/// of the way it runs, as `steps + 1` points.
fn bowed(a: (f32, f32), b: (f32, f32), bend: f32, steps: usize) -> Vec<(f32, f32)> {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let control = ((a.0 + b.0) * 0.5 + dy * bend, (a.1 + b.1) * 0.5 - dx * bend);
    (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32;
            let s = 1.0 - t;
            (
                s * s * a.0 + 2.0 * s * t * control.0 + t * t * b.0,
                s * s * a.1 + 2.0 * s * t * control.1 + t * t * b.1,
            )
        })
        .collect()
}

/// The first `fraction` of a polyline's length.
fn truncated(points: &[(f32, f32)], fraction: f32) -> Vec<(f32, f32)> {
    if fraction >= 1.0 || points.len() < 2 {
        return points.to_vec();
    }
    let lengths: Vec<f32> = points
        .windows(2)
        .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
        .collect();
    let mut left = lengths.iter().sum::<f32>() * fraction.max(0.0);
    let mut out = vec![points[0]];
    for (segment, length) in points.windows(2).zip(&lengths) {
        if left >= *length {
            out.push(segment[1]);
            left -= length;
        } else {
            let t = if *length > 0.0 { left / length } else { 0.0 };
            out.push((
                segment[0].0 + (segment[1].0 - segment[0].0) * t,
                segment[0].1 + (segment[1].1 - segment[0].1) * t,
            ));
            break;
        }
    }
    out
}

/// Coverage gathered over a box of the frame, blended once at the end.
struct Canvas {
    x0: i32,
    y0: i32,
    width: usize,
    height: usize,
    cover: Vec<f32>,
}

impl Canvas {
    fn over(frame: &Frame, points: &[(f32, f32)], margin: f32) -> Option<Canvas> {
        let (fw, fh) = (frame.width() as f32, frame.height() as f32);
        let x0 = points.iter().map(|p| p.0).fold(f32::MAX, f32::min) - margin;
        let x1 = points.iter().map(|p| p.0).fold(f32::MIN, f32::max) + margin;
        let y0 = points.iter().map(|p| p.1).fold(f32::MAX, f32::min) - margin;
        let y1 = points.iter().map(|p| p.1).fold(f32::MIN, f32::max) + margin;
        let (x0, x1) = (x0.floor().max(0.0), x1.ceil().min(fw));
        let (y0, y1) = (y0.floor().max(0.0), y1.ceil().min(fh));
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let (width, height) = ((x1 - x0) as usize, (y1 - y0) as usize);
        Some(Canvas {
            x0: x0 as i32,
            y0: y0 as i32,
            width,
            height,
            cover: vec![0.0; width * height],
        })
    }

    /// Every pixel of the canvas within `reach` of the box around `a`-`b`,
    /// with its centre in frame pixels.
    fn each(&mut self, lo: (f32, f32), hi: (f32, f32), mut f: impl FnMut(f32, f32, &mut f32)) {
        let xs = ((lo.0.floor() as i32 - self.x0).max(0) as usize)
            ..((hi.0.ceil() as i32 - self.x0 + 1).clamp(0, self.width as i32) as usize);
        let ys = ((lo.1.floor() as i32 - self.y0).max(0) as usize)
            ..((hi.1.ceil() as i32 - self.y0 + 1).clamp(0, self.height as i32) as usize);
        for y in ys {
            let py = (self.y0 + y as i32) as f32 + 0.5;
            for x in xs.clone() {
                let px = (self.x0 + x as i32) as f32 + 0.5;
                f(px, py, &mut self.cover[y * self.width + x]);
            }
        }
    }

    /// A thick line from `a` to `b`, dashed along `[from, to)` of the
    /// path's length when `dash` is set: `(on, off)` in pixels.
    fn segment(
        &mut self,
        a: (f32, f32),
        b: (f32, f32),
        half: f32,
        from: f32,
        dash: Option<(f32, f32)>,
    ) {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let length2 = (dx * dx + dy * dy).max(1e-6);
        let length = length2.sqrt();
        let lo = (a.0.min(b.0) - half - 1.0, a.1.min(b.1) - half - 1.0);
        let hi = (a.0.max(b.0) + half + 1.0, a.1.max(b.1) + half + 1.0);
        self.each(lo, hi, |px, py, cover| {
            let t = (((px - a.0) * dx + (py - a.1) * dy) / length2).clamp(0.0, 1.0);
            let (qx, qy) = (a.0 + dx * t, a.1 + dy * t);
            let distance = (px - qx).hypot(py - qy);
            let mut c = (half - distance + 0.5).clamp(0.0, 1.0);
            if let Some((on, off)) = dash {
                let along = (from + t * length) % (on + off);
                if along > on {
                    c = 0.0;
                }
            }
            *cover = cover.max(c);
        });
    }

    /// A filled triangle, its edges a pixel soft.
    fn triangle(&mut self, p: [(f32, f32); 3]) {
        let lo = (
            p.iter().map(|q| q.0).fold(f32::MAX, f32::min) - 1.0,
            p.iter().map(|q| q.1).fold(f32::MAX, f32::min) - 1.0,
        );
        let hi = (
            p.iter().map(|q| q.0).fold(f32::MIN, f32::max) + 1.0,
            p.iter().map(|q| q.1).fold(f32::MIN, f32::max) + 1.0,
        );
        let area = (p[1].0 - p[0].0) * (p[2].1 - p[0].1) - (p[2].0 - p[0].0) * (p[1].1 - p[0].1);
        let sign = if area < 0.0 { -1.0 } else { 1.0 };
        self.each(lo, hi, |px, py, cover| {
            // The distance inside the nearest edge, positive within.
            let inside = (0..3)
                .map(|i| {
                    let (a, b) = (p[i], p[(i + 1) % 3]);
                    let (ex, ey) = (b.0 - a.0, b.1 - a.1);
                    let length = ex.hypot(ey).max(1e-6);
                    sign * (ex * (py - a.1) - ey * (px - a.0)) / length
                })
                .fold(f32::MAX, f32::min);
            *cover = cover.max((inside + 0.5).clamp(0.0, 1.0));
        });
    }

    /// A filled disc.
    fn disc(&mut self, centre: (f32, f32), radius: f32) {
        let lo = (centre.0 - radius - 1.0, centre.1 - radius - 1.0);
        let hi = (centre.0 + radius + 1.0, centre.1 + radius + 1.0);
        self.each(lo, hi, |px, py, cover| {
            let distance = (px - centre.0).hypot(py - centre.1);
            *cover = cover.max((radius - distance + 0.5).clamp(0.0, 1.0));
        });
    }

    /// A polygon filled to `level`, by the even-odd rule on pixel centres.
    fn fill(&mut self, polygon: &[(f32, f32)], level: f32) {
        let (x0, y0, width) = (self.x0, self.y0, self.width);
        for y in 0..self.height {
            let py = (y0 + y as i32) as f32 + 0.5;
            let mut crossings: Vec<f32> = Vec::new();
            for i in 0..polygon.len() {
                let (a, b) = (polygon[i], polygon[(i + 1) % polygon.len()]);
                if (a.1 <= py) != (b.1 <= py) {
                    crossings.push(a.0 + (py - a.1) / (b.1 - a.1) * (b.0 - a.0));
                }
            }
            crossings.sort_by(|a, b| a.total_cmp(b));
            for pair in crossings.chunks_exact(2) {
                let from = ((pair[0] - 0.5).ceil() as i32 - x0).clamp(0, width as i32) as usize;
                let to = ((pair[1] - 0.5).floor() as i32 - x0 + 1).clamp(0, width as i32) as usize;
                for x in from..to {
                    let cover = &mut self.cover[y * width + x];
                    *cover = cover.max(level);
                }
            }
        }
    }

    /// The gathered coverage painted onto the frame in `colour`, `alpha`
    /// solid at full coverage.
    fn paint(&self, frame: &mut Frame, colour: [u8; 3], alpha: f32) {
        let width = frame.width() as usize;
        let pixels = frame.pixels_mut();
        for y in 0..self.height {
            for x in 0..self.width {
                let mix = self.cover[y * self.width + x] * alpha;
                if mix <= 0.0 {
                    continue;
                }
                let at = ((self.y0 as usize + y) * width + self.x0 as usize + x) * BYTES_PER_PIXEL;
                for c in 0..3 {
                    pixels[at + c] = (f32::from(pixels[at + c]) * (1.0 - mix)
                        + f32::from(colour[c]) * mix
                        + 0.5) as u8;
                }
            }
        }
    }
}

/// Draws `sketch` onto `frame`.
pub fn draw(frame: &mut Frame, sketch: &Sketch, mapping: &Mapping) {
    if sketch.points.len() < 2 || sketch.opacity <= 0.0 || sketch.reveal <= 0.0 {
        return;
    }
    // The line's half width: three pixels at 720 lines, scaled with the
    // picture and the knob.
    let half = (frame.height() as f32 / 240.0 * sketch.thickness).max(0.75);
    let mut path: Vec<(f32, f32)> = sketch
        .points
        .iter()
        .map(|&point| to_pixels(mapping, frame, point))
        .collect();
    if sketch.kind == Kind::Curve {
        path = bowed(path[0], path[path.len() - 1], sketch.bend, 32);
    }
    if sketch.kind == Kind::Zone {
        path.push(path[0]);
    }
    let headed = matches!(sketch.kind, Kind::Arrow | Kind::Curve);
    let head_long = half * 5.0;
    let head_half = half * 3.2;
    let shown = if sketch.kind == Kind::Zone {
        path.clone()
    } else {
        truncated(&path, sketch.reveal)
    };
    let Some(mut canvas) = Canvas::over(frame, &path, head_long + half + 2.0) else {
        return;
    };
    // The shaft stops short of the tip, where the head takes over.
    let mut shaft = shown.clone();
    let mut head: Option<[(f32, f32); 3]> = None;
    if headed && shown.len() >= 2 {
        let tip = shown[shown.len() - 1];
        // The way the line comes in, over the last head's length of it.
        let mut back = shown[shown.len() - 2];
        for point in shown.iter().rev().skip(1) {
            back = *point;
            if (tip.0 - point.0).hypot(tip.1 - point.1) >= head_long {
                break;
            }
        }
        let (dx, dy) = (tip.0 - back.0, tip.1 - back.1);
        let length = dx.hypot(dy);
        if length > 1e-3 {
            let (ux, uy) = (dx / length, dy / length);
            let base = (tip.0 - ux * head_long, tip.1 - uy * head_long);
            head = Some([
                tip,
                (base.0 - uy * head_half, base.1 + ux * head_half),
                (base.0 + uy * head_half, base.1 - ux * head_half),
            ]);
            if let Some(last) = shaft.last_mut() {
                *last = (tip.0 - ux * head_long * 0.6, tip.1 - uy * head_long * 0.6);
            }
        }
    }
    let dash = sketch.dashed.then_some((half * 6.0, half * 4.0));
    let mut walked = 0.0;
    for pair in shaft.windows(2) {
        canvas.segment(pair[0], pair[1], half, walked, dash);
        walked += (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1);
    }
    if let Some(head) = head {
        canvas.triangle(head);
    }
    if sketch.kind == Kind::Link {
        // A dot on every player the line joins, as far as it has reached.
        let reached = (sketch.reveal * (path.len() - 1) as f32).floor() as usize;
        for point in path.iter().take(reached + 1) {
            canvas.disc(*point, half * 2.6);
        }
    }
    if sketch.kind == Kind::Zone {
        // The area under a faint wash, fading in as the drawing appears.
        canvas.fill(&path[..path.len() - 1], 0.28 * sketch.reveal.min(1.0));
    }
    canvas.paint(frame, sketch.colour, sketch.opacity.clamp(0.0, 1.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grass() -> Frame {
        let mut frame = Frame::black(200, 100);
        frame.fill([40, 140, 50, 255]);
        frame
    }

    fn sketch(kind: Kind, points: Vec<(f32, f32)>) -> Sketch {
        Sketch {
            kind,
            points,
            dashed: false,
            colour: [255, 0, 0],
            thickness: 1.0,
            bend: 0.25,
            reveal: 1.0,
            opacity: 1.0,
        }
    }

    fn red(frame: &Frame, x: u32, y: u32) -> bool {
        let p = frame.pixel(x, y).unwrap();
        p[0] > 200 && p[1] < 60
    }

    #[test]
    fn a_line_is_drawn_between_its_points_and_nowhere_else() {
        let mut frame = grass();
        draw(
            &mut frame,
            &sketch(Kind::Line, vec![(0.1, 0.5), (0.9, 0.5)]),
            &Mapping::IDENTITY,
        );
        assert!(red(&frame, 100, 50));
        assert!(!red(&frame, 100, 20));
        assert!(!red(&frame, 5, 50));
    }

    #[test]
    fn an_arrow_has_a_head_wider_than_its_shaft() {
        let mut frame = Frame::black(720, 360);
        frame.fill([40, 140, 50, 255]);
        let mut arrow = sketch(Kind::Arrow, vec![(0.1, 0.5), (0.9, 0.5)]);
        arrow.thickness = 2.0;
        draw(&mut frame, &arrow, &Mapping::IDENTITY);
        let column = |x: u32| (0..360).filter(|&y| red(&frame, x, y)).count();
        // The head is the last fifteen pixels before the tip at 648.
        assert!(
            column(636) > column(300) + 4,
            "{} {}",
            column(636),
            column(300)
        );
    }

    #[test]
    fn a_half_revealed_line_stops_half_way() {
        let mut frame = grass();
        let mut half = sketch(Kind::Line, vec![(0.1, 0.5), (0.9, 0.5)]);
        half.reveal = 0.5;
        draw(&mut frame, &half, &Mapping::IDENTITY);
        assert!(red(&frame, 60, 50));
        assert!(!red(&frame, 150, 50));
    }

    #[test]
    fn a_dashed_line_has_gaps() {
        let mut frame = grass();
        let mut dashed = sketch(Kind::Line, vec![(0.0, 0.5), (1.0, 0.5)]);
        dashed.dashed = true;
        draw(&mut frame, &dashed, &Mapping::IDENTITY);
        let lit = (0..200).filter(|&x| red(&frame, x, 50)).count();
        assert!(lit > 60 && lit < 180, "{lit}");
    }

    #[test]
    fn a_zone_washes_its_inside() {
        let mut frame = grass();
        draw(
            &mut frame,
            &sketch(
                Kind::Zone,
                vec![(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            ),
            &Mapping::IDENTITY,
        );
        let inside = frame.pixel(100, 50).unwrap();
        assert!(inside[0] > 40 && inside[0] < 200, "{inside:?}");
        assert!(red(&frame, 40, 50));
    }

    #[test]
    fn a_curve_bows_away_from_the_straight_line() {
        let mut frame = grass();
        draw(
            &mut frame,
            &sketch(Kind::Curve, vec![(0.1, 0.5), (0.9, 0.5)]),
            &Mapping::IDENTITY,
        );
        assert!(!red(&frame, 100, 50));
        let bowed = (0..100).any(|y| red(&frame, 100, y));
        assert!(bowed);
    }
}
