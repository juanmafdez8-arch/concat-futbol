// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! A clip's football marks, drawn on its decoded frame.
//!
//! Every mark is about one tracked thing at one instant: its box, and its
//! silhouette when the brush model has read one. Without a silhouette the
//! thing is a soft ellipse inside its box, which is what a player looks
//! like from a broadcast camera to within a few pixels, so a mark works the
//! moment the analysis has run and only gets sharper when the silhouettes
//! arrive.
//!
//! The frame is the source after the clip's crop and flips, fitted to some
//! size; a box or a silhouette is in source fractions. [`Mapping`] walks a
//! frame pixel back to the source, exactly as a cutout does, so the marks
//! sit where the cutout would.

use concat_core::frame::{BYTES_PER_PIXEL, Frame};

use crate::{Mapping, Mask};

/// How a mark is shaped against its default: see the document's
/// `FootballMark` for what each knob does to each mark.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    /// Overall size.
    pub size: f32,
    /// Weight of the line.
    pub thickness: f32,
    /// Length: the height of the round marks and the frame, the arrow's
    /// length.
    pub squash: f32,
    /// Width: the round marks', the frame's and the arrow's, on its own.
    pub width: f32,
}

impl Shape {
    /// Every knob at its default.
    pub const DEFAULT: Shape = Shape {
        size: 1.0,
        thickness: 1.0,
        squash: 1.0,
        width: 1.0,
    };
}

impl Default for Shape {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The thing a mark is drawn about, at one instant.
#[derive(Clone, Copy)]
pub struct Target<'a> {
    /// Its box, as source fractions `[left, top, right, bottom]`.
    pub bbox: [f32; 4],
    /// Its silhouette over the whole source, if one has been read.
    pub silhouette: Option<&'a Mask>,
    /// How far the thing has moved since the silhouette's instant, in
    /// source fractions: the silhouette is read this much behind.
    pub shift: (f32, f32),
}

impl Target<'_> {
    /// How much a source point is the thing, `0..=1`.
    pub fn weight(&self, u: f32, v: f32) -> f32 {
        let [l, t, r, b] = self.bbox;
        let (w, h) = ((r - l).max(1e-4), (b - t).max(1e-4));
        // Nothing outside the box, grown a little for an arm or a boot.
        if u < l - w * 0.25 || u > r + w * 0.25 || v < t - h * 0.1 || v > b + h * 0.1 {
            return 0.0;
        }
        match self.silhouette {
            Some(mask) => mask.sample(u - self.shift.0, v - self.shift.1),
            None => self.ellipse(u, v, 1.0),
        }
    }

    /// The soft ellipse filling the box, `scale` times its size: the thing
    /// without a silhouette.
    fn ellipse(&self, u: f32, v: f32, scale: f32) -> f32 {
        let [l, t, r, b] = self.bbox;
        let (w, h) = ((r - l).max(1e-4), (b - t).max(1e-4));
        let (cx, cy) = ((l + r) * 0.5, (t + b) * 0.5);
        let dx = (u - cx) / (w * 0.6 * scale);
        let dy = (v - cy) / (h * 0.58 * scale);
        let d = (dx * dx + dy * dy).sqrt();
        1.0 - smoothstep(0.8, 1.0, d)
    }

    /// What a hole must cover: the silhouette and the whole box with it,
    /// `reach` times as big, since a boot or a shadow the silhouette
    /// missed left behind reads worse than a little grass filled for
    /// nothing.
    fn hole(&self, u: f32, v: f32, reach: f32) -> f32 {
        self.weight(u, v).max(self.ellipse(u, v, reach))
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The frame pixels a source box covers, grown by `margin` pixels and
/// held to the frame: `(x0, y0, x1, y1)`, end exclusive.
fn pixel_box(
    frame: &Frame,
    bbox: &[f32; 4],
    mapping: &Mapping,
    margin: f32,
) -> Option<(usize, usize, usize, usize)> {
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    let corners = [
        (bbox[0], bbox[1]),
        (bbox[2], bbox[1]),
        (bbox[0], bbox[3]),
        (bbox[2], bbox[3]),
    ];
    // A corner the crop cut away is held to the crop's edge.
    let [cl, ct, cr, cb] = mapping.crop;
    let held: Vec<(f32, f32)> = corners
        .iter()
        .filter_map(|&(u, v)| {
            let u = u.clamp(cl, 1.0 - cr);
            let v = v.clamp(ct, 1.0 - cb);
            mapping.frame_of(u, v)
        })
        .collect();
    if held.is_empty() {
        return None;
    }
    let x0 = held.iter().map(|p| p.0).fold(f32::MAX, f32::min) * fw - margin;
    let x1 = held.iter().map(|p| p.0).fold(f32::MIN, f32::max) * fw + margin;
    let y0 = held.iter().map(|p| p.1).fold(f32::MAX, f32::min) * fh - margin;
    let y1 = held.iter().map(|p| p.1).fold(f32::MIN, f32::max) * fh + margin;
    let x0 = x0.floor().clamp(0.0, fw) as usize;
    let x1 = x1.ceil().clamp(0.0, fw) as usize;
    let y0 = y0.floor().clamp(0.0, fh) as usize;
    let y1 = y1.ceil().clamp(0.0, fh) as usize;
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
}

/// Calls `paint` for every frame pixel in the box with its source point.
///
/// Rows are split across the machine's cores once the box is worth the
/// threads, the way [`crate::cut`] does: a spotlight or an isolated layer
/// touches every pixel of every frame, and on a 4K timeline one core is
/// not enough to keep the monitor playing.
fn each_in(
    frame: &mut Frame,
    region: (usize, usize, usize, usize),
    mapping: &Mapping,
    paint: impl Fn(&mut [u8], f32, f32) + Sync,
) {
    let (width, height) = (frame.width() as usize, frame.height() as usize);
    let (x0, y0, x1, y1) = region;
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let row_bytes = width * BYTES_PER_PIXEL;
    let rows = &mut frame.pixels_mut()[y0 * row_bytes..y1 * row_bytes];
    let workers = if (x1 - x0) * (y1 - y0) >= 200_000 {
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .clamp(1, 8)
    } else {
        1
    };
    let per = (y1 - y0).div_ceil(workers);
    let paint = &paint;
    let band = |first: usize, chunk: &mut [u8]| {
        for (offset, row) in chunk.chunks_exact_mut(row_bytes).enumerate() {
            let fy = ((first + offset) as f32 + 0.5) / height as f32;
            for x in x0..x1 {
                let fx = (x as f32 + 0.5) / width as f32;
                let (u, v) = mapping.source_of(fx, fy);
                let at = x * BYTES_PER_PIXEL;
                paint(&mut row[at..at + BYTES_PER_PIXEL], u, v);
            }
        }
    };
    if workers == 1 {
        band(y0, rows);
        return;
    }
    std::thread::scope(|scope| {
        for (index, chunk) in rows.chunks_mut(per * row_bytes).enumerate() {
            scope.spawn(move || band(y0 + index * per, chunk));
        }
    });
}

/// The thing's silhouette tinted `colour`: the shading an analyst draws
/// over a player to say "this one".
pub fn shade(
    frame: &mut Frame,
    target: &Target<'_>,
    mapping: &Mapping,
    colour: [u8; 3],
    strength: f32,
) {
    let Some(region) = pixel_box(frame, &target.bbox, mapping, 4.0) else {
        return;
    };
    let strength = strength.clamp(0.0, 1.0) * 0.75;
    each_in(frame, region, mapping, |pixel, u, v| {
        let mix = target.weight(u, v) * strength;
        if mix > 0.0 {
            for c in 0..3 {
                pixel[c] =
                    (f32::from(pixel[c]) * (1.0 - mix) + f32::from(colour[c]) * mix + 0.5) as u8;
            }
        }
    });
}

/// The picture darkened outside a soft pool of light around each of the
/// things, in one pass: two players spotlit share one darkness, rather
/// than each darkening the other's pool. `strength` is the strongest of
/// the marks asking.
pub fn spotlight(
    frame: &mut Frame,
    targets: &[(Target<'_>, Shape)],
    mapping: &Mapping,
    aspect: f32,
    strength: f32,
) {
    let strength = strength.clamp(0.0, 1.0);
    if targets.is_empty() || strength == 0.0 {
        return;
    }
    // A pool a little wider than the player is tall, round on screen,
    // scaled by its size and stood up or laid down by its squash.
    let pools: Vec<(f32, f32, f32, f32, f32, [f32; 4])> = targets
        .iter()
        .map(|(target, shape)| {
            let [l, t, r, b] = target.bbox;
            let radius = ((b - t).max((r - l) * aspect) * 0.9 + 0.02) * shape.size;
            (
                (l + r) * 0.5,
                (t + b) * 0.5,
                radius,
                shape.squash.max(0.05),
                shape.width.max(0.05),
                [
                    (l + r) * 0.5 - radius * shape.width.max(0.05) / aspect.max(1e-4),
                    (t + b) * 0.5 - radius * shape.squash.max(0.05),
                    (l + r) * 0.5 + radius * shape.width.max(0.05) / aspect.max(1e-4),
                    (t + b) * 0.5 + radius * shape.squash.max(0.05),
                ],
            )
        })
        .collect();
    let dim = 1.0 - strength * 0.7;
    let full = (0, 0, frame.width() as usize, frame.height() as usize);
    each_in(frame, full, mapping, |pixel, u, v| {
        let lit = pools
            .iter()
            .filter(|pool| {
                let bounds = pool.5;
                u >= bounds[0] && u <= bounds[2] && v >= bounds[1] && v <= bounds[3]
            })
            .map(|&(cx, cy, radius, squash, wide, _)| {
                let d = ((u - cx) * aspect / wide).hypot((v - cy) / squash) / radius;
                1.0 - smoothstep(0.75, 1.0, d)
            })
            .fold(0.0, f32::max);
        let gain = dim + (1.0 - dim) * lit;
        if gain < 0.999 {
            for channel in pixel.iter_mut().take(3) {
                *channel = (f32::from(*channel) * gain + 0.5) as u8;
            }
        }
    });
}

/// A ring on the grass round the thing's feet, drawn flattened as the
/// pitch is seen from the stands.
pub fn ring(
    frame: &mut Frame,
    target: &Target<'_>,
    mapping: &Mapping,
    aspect: f32,
    colour: [u8; 3],
    strength: f32,
    shape: Shape,
) {
    let [l, t, r, b] = target.bbox;
    let height = b - t;
    let cx = (l + r) * 0.5;
    let cy = b - height * 0.02;
    // Radii in source fractions: a little wider than a stride, and a
    // third as tall, which is how a circle on the pitch looks from a
    // broadcast gantry.
    let across = ((r - l) * 0.9).max(height * 0.45 / aspect) * shape.size;
    let rx = across * shape.width;
    let ry = across * aspect * 0.32 * shape.squash;
    let thick = (0.18 * shape.thickness).clamp(0.03, 0.9);
    let ring_box = [cx - rx * 1.3, cy - ry * 1.6, cx + rx * 1.3, cy + ry * 1.6];
    let Some(region) = pixel_box(frame, &ring_box, mapping, 2.0) else {
        return;
    };
    let strength = strength.clamp(0.0, 1.0);
    each_in(frame, region, mapping, |pixel, u, v| {
        let d = ((u - cx) / rx).hypot((v - cy) / ry);
        let edge = (d - 1.0).abs();
        let mut mix = (1.0 - smoothstep(thick * 0.5, thick, edge)) * strength;
        // A faint fill inside the ring.
        if d < 1.0 {
            mix = mix.max(0.18 * strength);
        }
        // The player stands in front of the ring's far half.
        if v < cy && target.weight(u, v) > 0.5 {
            mix = 0.0;
        }
        if mix > 0.0 {
            for c in 0..3 {
                pixel[c] =
                    (f32::from(pixel[c]) * (1.0 - mix) + f32::from(colour[c]) * mix + 0.5) as u8;
            }
        }
    });
}

/// The thing taken out of the picture: the hole it leaves is filled from
/// what surrounds it, pulled in from the edge at ever coarser scales and
/// pushed back down (pull-push), which on grass reads as grass.
pub fn erase(frame: &mut Frame, target: &Target<'_>, mapping: &Mapping, reach: f32) {
    let (fw, fh) = (frame.width() as usize, frame.height() as usize);
    let reach = reach.max(0.25);
    let hole_box = super::grown(&target.bbox, ((reach - 1.0) * 0.6).max(0.0));
    let Some((x0, y0, x1, y1)) = pixel_box(frame, &hole_box, mapping, 0.0) else {
        return;
    };
    // Work on a patch twice the player's size, so there is plenty of
    // picture round the hole to fill it from.
    let pad_x = ((x1 - x0) as f32 * 0.6).max(12.0) as usize;
    let pad_y = ((y1 - y0) as f32 * 0.4).max(12.0) as usize;
    let (px0, py0) = (x0.saturating_sub(pad_x), y0.saturating_sub(pad_y));
    let (px1, py1) = ((x1 + pad_x).min(fw), (y1 + pad_y).min(fh));
    let (pw, ph) = (px1 - px0, py1 - py0);
    if pw < 2 || ph < 2 {
        return;
    }

    // The hole, grown by a couple of pixels so no edge of the player is
    // left behind, and how much each pixel is it.
    let mut weight = vec![0f32; pw * ph];
    for y in 0..ph {
        let fy = ((py0 + y) as f32 + 0.5) / fh as f32;
        for x in 0..pw {
            let fx = ((px0 + x) as f32 + 0.5) / fw as f32;
            let (u, v) = mapping.source_of(fx, fy);
            weight[y * pw + x] = target.hole(u, v, reach);
        }
    }
    let grow = ((x1 - x0).max(y1 - y0) as f32 * 0.06).clamp(1.0, 6.0) as usize;
    let weight = dilate(&weight, pw, ph, grow);

    let pixels = frame.pixels();
    let mut colour = vec![[0f32; 3]; pw * ph];
    let mut known = vec![0f32; pw * ph];
    for y in 0..ph {
        for x in 0..pw {
            let at = ((py0 + y) * fw + px0 + x) * BYTES_PER_PIXEL;
            let i = y * pw + x;
            colour[i] = [
                f32::from(pixels[at]),
                f32::from(pixels[at + 1]),
                f32::from(pixels[at + 2]),
            ];
            known[i] = if weight[i] > 0.1 { 0.0 } else { 1.0 };
        }
    }
    let filled = pull_push(&colour, &known, pw, ph);

    // A little of the grass's own grain back, so the fill is not a smear.
    let grain = local_grain(&colour, &known);
    let pixels = frame.pixels_mut();
    for y in 0..ph {
        for x in 0..pw {
            let i = y * pw + x;
            let mix = weight[i].clamp(0.0, 1.0);
            if mix <= 0.0 {
                continue;
            }
            // Grain two pixels across, as compressed grass is, and faint.
            let noise = (hash((px0 + x) / 2, (py0 + y) / 2) - 0.5) * grain * 0.6;
            let at = ((py0 + y) * fw + px0 + x) * BYTES_PER_PIXEL;
            for c in 0..3 {
                let fill = (filled[i][c] + noise).clamp(0.0, 255.0);
                let was = f32::from(pixels[at + c]);
                pixels[at + c] = (was * (1.0 - mix) + fill * mix + 0.5) as u8;
            }
        }
    }
}

/// An arrow over the thing's head, pointing down at it, the way a
/// broadcast picks a player out.
pub fn arrow(
    frame: &mut Frame,
    target: &Target<'_>,
    mapping: &Mapping,
    aspect: f32,
    colour: [u8; 3],
    strength: f32,
    shape: Shape,
) {
    let [l, t, r, b] = target.bbox;
    let height = (b - t).max(1e-4);
    let cx = (l + r) * 0.5;
    // The tip a little above the head; the arrow as long as half the
    // player, and as wide as it is long, on screen.
    let tip = t - height * 0.12;
    let long = height * 0.45 * shape.size * shape.squash;
    let half = height * 0.22 * shape.size * shape.width / aspect.max(1e-4);
    let arrow_box = [cx - half, tip - long, cx + half, tip];
    let Some(region) = pixel_box(frame, &arrow_box, mapping, 2.0) else {
        return;
    };
    let strength = strength.clamp(0.0, 1.0);
    // A pixel's width in source fractions, for a soft edge one pixel wide.
    let soft = 1.0 / frame.width().max(1) as f32;
    let outline = 1.5 * soft * shape.thickness;
    each_in(frame, region, mapping, |pixel, u, v| {
        let along = (tip - v) / long;
        if !(0.0..=1.0).contains(&along) {
            return;
        }
        let inside = half * along - (u - cx).abs();
        if inside < -soft {
            return;
        }
        // Filled in the colour, with a dark rim so it reads on grass and
        // on a white shirt alike.
        let cover = smoothstep(-soft, soft, inside) * strength;
        let rim = 1.0 - smoothstep(outline, outline + soft, inside);
        for c in 0..3 {
            let paint = f32::from(colour[c]) * (1.0 - rim * 0.6);
            pixel[c] = (f32::from(pixel[c]) * (1.0 - cover) + paint * cover + 0.5) as u8;
        }
    });
}

/// A frame drawn round the thing: a rectangle a little bigger than it,
/// its line as heavy as `shape.thickness` says, stretched across by
/// `shape.width` and up and down by `shape.squash`.
pub fn frame_round(
    frame: &mut Frame,
    target: &Target<'_>,
    mapping: &Mapping,
    colour: [u8; 3],
    strength: f32,
    shape: Shape,
) {
    let grown = stretched(&super::grown(&target.bbox, 0.12 * shape.size), shape);
    let Some((x0, y0, x1, y1)) = pixel_box(frame, &grown, mapping, 0.0) else {
        return;
    };
    // Two pixels at 720 lines, scaled with the picture and the knob.
    let line = ((frame.height() as f32 / 360.0) * shape.thickness)
        .round()
        .max(1.0) as usize;
    let strength = strength.clamp(0.0, 1.0);
    let width = frame.width() as usize;
    let pixels = frame.pixels_mut();
    for y in y0..y1 {
        for x in x0..x1 {
            let edge = x < x0 + line || x + line >= x1 || y < y0 + line || y + line >= y1;
            if edge {
                let at = (y * width + x) * BYTES_PER_PIXEL;
                for c in 0..3 {
                    pixels[at + c] = (f32::from(pixels[at + c]) * (1.0 - strength)
                        + f32::from(colour[c]) * strength
                        + 0.5) as u8;
                }
            }
        }
    }
}

/// A box scaled about its centre: across by the shape's width, up and down
/// by its length.
fn stretched(bbox: &[f32; 4], shape: Shape) -> [f32; 4] {
    let [l, t, r, b] = *bbox;
    let (cx, cy) = ((l + r) * 0.5, (t + b) * 0.5);
    let (hw, hh) = ((r - l) * 0.5 * shape.width, (b - t) * 0.5 * shape.squash);
    [cx - hw, cy - hh, cx + hw, cy + hh]
}

/// A magnifier: a disc beside the thing showing it `zoom` times larger,
/// ringed in `colour`. It sits up and to the right of the player, or to
/// the left when that would leave the picture, so the player stays in
/// view under it.
pub fn lens(
    frame: &mut Frame,
    target: &Target<'_>,
    mapping: &Mapping,
    colour: [u8; 3],
    strength: f32,
    shape: Shape,
) {
    let Some((x0, y0, x1, y1)) = pixel_box(frame, &target.bbox, mapping, 0.0) else {
        return;
    };
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    let (cx, cy) = ((x0 + x1) as f32 * 0.5, (y0 + y1) as f32 * 0.5);
    let tall = (y1 - y0).max(1) as f32;
    let radius = (tall * 1.1 * shape.size).clamp(fh * 0.06, fh * 0.3);
    let zoom = 2.2 * shape.squash.max(0.25);
    let ring = (fh / 240.0 * shape.thickness).max(1.5);
    let mut lx = cx + tall * 0.6 + radius;
    if lx + radius > fw {
        lx = cx - tall * 0.6 - radius;
    }
    let ly = (cy - tall * 0.4 - radius * 0.6).clamp(radius, fh - radius);
    let source = frame.clone();
    let width = frame.width() as usize;
    let pixels = frame.pixels_mut();
    let strength = strength.clamp(0.0, 1.0);
    let (ax, ay) = ((lx - radius).max(0.0) as usize, (ly - radius).max(0.0) as usize);
    let (bx, by) = (
        ((lx + radius + ring) as usize).min(width),
        ((ly + radius + ring) as usize).min(source.height() as usize),
    );
    for y in ay..by {
        for x in ax..bx {
            let (dx, dy) = (x as f32 + 0.5 - lx, y as f32 + 0.5 - ly);
            let d = dx.hypot(dy);
            if d > radius + ring {
                continue;
            }
            let at = (y * width + x) * BYTES_PER_PIXEL;
            if d > radius {
                for c in 0..3 {
                    pixels[at + c] = (f32::from(pixels[at + c]) * (1.0 - strength)
                        + f32::from(colour[c]) * strength
                        + 0.5) as u8;
                }
                continue;
            }
            let sx = (cx + dx / zoom).clamp(0.0, fw - 1.0) as u32;
            let sy = (cy + dy / zoom).clamp(0.0, fh - 1.0) as u32;
            if let Some(p) = source.pixel(sx, sy) {
                pixels[at..at + 3].copy_from_slice(&p[..3]);
            }
        }
    }
}

/// Everything but the thing made transparent: the layer a moved or a
/// pasted player is.
pub fn isolate(frame: &mut Frame, target: &Target<'_>, mapping: &Mapping) {
    // Only the player's box is weighed; everything outside it is gone,
    // which is a fill, not a lookup per pixel.
    let region = pixel_box(frame, &super::grown(&target.bbox, 0.3), mapping, 2.0);
    let (width, height) = (frame.width() as usize, frame.height() as usize);
    let (x0, y0, x1, y1) = region.unwrap_or((0, 0, 0, 0));
    let pixels = frame.pixels_mut();
    let row_bytes = width * BYTES_PER_PIXEL;
    for y in 0..height {
        let row = &mut pixels[y * row_bytes..(y + 1) * row_bytes];
        if y < y0 || y >= y1 {
            for pixel in row.chunks_exact_mut(BYTES_PER_PIXEL) {
                pixel[3] = 0;
            }
        } else {
            for pixel in row[..x0 * BYTES_PER_PIXEL].chunks_exact_mut(BYTES_PER_PIXEL) {
                pixel[3] = 0;
            }
            for pixel in row[x1 * BYTES_PER_PIXEL..].chunks_exact_mut(BYTES_PER_PIXEL) {
                pixel[3] = 0;
            }
        }
    }
    if region.is_some() && height > 0 {
        each_in(frame, (x0, y0, x1, y1), mapping, |pixel, u, v| {
            let keep = target.weight(u, v);
            pixel[3] = (f32::from(pixel[3]) * keep + 0.5) as u8;
        });
    }
}

/// Every tracked box outlined, the chosen one thicker and in `chosen`'s
/// colour: what the monitor shows while a player is being picked.
pub fn outline(frame: &mut Frame, boxes: &[([f32; 4], [u8; 3], bool)], mapping: &Mapping) {
    for (bbox, colour, chosen) in boxes {
        let Some((x0, y0, x1, y1)) = pixel_box(frame, bbox, mapping, 1.0) else {
            continue;
        };
        let width = frame.width() as usize;
        let line = if *chosen { 3 } else { 1 };
        let pixels = frame.pixels_mut();
        for y in y0..y1 {
            for x in x0..x1 {
                let edge = x < x0 + line || x + line >= x1 || y < y0 + line || y + line >= y1;
                if edge {
                    let at = (y * width + x) * BYTES_PER_PIXEL;
                    pixels[at..at + 3].copy_from_slice(colour);
                    pixels[at + 3] = 255;
                }
            }
        }
    }
}

/// The largest weight within `radius` pixels, in a square: the hole grown.
fn dilate(weight: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut rows = vec![0f32; weight.len()];
    for y in 0..height {
        for x in 0..width {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(width);
            rows[y * width + x] = weight[y * width + lo..y * width + hi]
                .iter()
                .copied()
                .fold(0.0, f32::max);
        }
    }
    let mut out = vec![0f32; weight.len()];
    for y in 0..height {
        let lo = y.saturating_sub(radius);
        let hi = (y + radius + 1).min(height);
        for x in 0..width {
            out[y * width + x] = (lo..hi).map(|yy| rows[yy * width + x]).fold(0.0, f32::max);
        }
    }
    out
}

/// Pull-push hole filling: the known pixels averaged down a pyramid until
/// every level is covered, then pushed back up into the holes.
fn pull_push(colour: &[[f32; 3]], known: &[f32], width: usize, height: usize) -> Vec<[f32; 3]> {
    if width <= 1 && height <= 1 {
        return colour.to_vec();
    }
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let mut coarse_colour = vec![[0f32; 3]; cw * ch];
    let mut coarse_known = vec![0f32; cw * ch];
    for y in 0..ch {
        for x in 0..cw {
            let mut sum = [0f32; 3];
            let mut weight = 0f32;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (sx, sy) = (2 * x + dx, 2 * y + dy);
                if sx < width && sy < height {
                    let i = sy * width + sx;
                    let k = known[i];
                    for c in 0..3 {
                        sum[c] += colour[i][c] * k;
                    }
                    weight += k;
                }
            }
            let i = y * cw + x;
            if weight > 0.0 {
                coarse_colour[i] = sum.map(|s| s / weight);
            }
            coarse_known[i] = weight.min(1.0);
        }
    }
    let coarse = if coarse_known.iter().all(|&k| k >= 1.0) || (cw <= 1 && ch <= 1) {
        coarse_colour
    } else {
        pull_push(&coarse_colour, &coarse_known, cw, ch)
    };
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| {
            let i = y * width + x;
            let up = bilinear_rgb(
                &coarse,
                cw,
                ch,
                (x as f32 - 0.5) * 0.5,
                (y as f32 - 0.5) * 0.5,
            );
            let k = known[i];
            std::array::from_fn(|c| colour[i][c] * k + up[c] * (1.0 - k))
        })
        .collect()
}

fn bilinear_rgb(plane: &[[f32; 3]], width: usize, height: usize, x: f32, y: f32) -> [f32; 3] {
    let at = |x: i64, y: i64| {
        let x = x.clamp(0, width as i64 - 1) as usize;
        let y = y.clamp(0, height as i64 - 1) as usize;
        plane[y * width + x]
    };
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let (a, b, c, d) = (
        at(x0, y0),
        at(x0 + 1, y0),
        at(x0, y0 + 1),
        at(x0 + 1, y0 + 1),
    );
    std::array::from_fn(|i| {
        let top = a[i] * (1.0 - fx) + b[i] * fx;
        let bottom = c[i] * (1.0 - fx) + d[i] * fx;
        top * (1.0 - fy) + bottom * fy
    })
}

/// The known pixels' brightness spread: how grainy the surroundings are.
fn local_grain(colour: &[[f32; 3]], known: &[f32]) -> f32 {
    let lumas: Vec<f32> = colour
        .iter()
        .zip(known)
        .filter(|(_, k)| **k >= 1.0)
        .map(|(c, _)| 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
        .collect();
    if lumas.len() < 2 {
        return 0.0;
    }
    let mean = lumas.iter().sum::<f32>() / lumas.len() as f32;
    let var = lumas.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / lumas.len() as f32;
    // Other players and the lines in the patch are not grain; held low.
    var.sqrt().min(5.0)
}

/// A steady pseudo-random value in `0..1` for a pixel, so the grain does
/// not crawl from frame to frame.
fn hash(x: usize, y: usize) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9e37_79b1) ^ (y as u32).wrapping_mul(0x85eb_ca77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    (h & 0xffff) as f32 / 65535.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grass_with_player() -> Frame {
        let mut frame = Frame::black(64, 64);
        frame.fill([40, 140, 50, 255]);
        for y in 20..44 {
            for x in 28..36 {
                frame.set_pixel(x, y, [230, 30, 30, 255]);
            }
        }
        frame
    }

    fn target() -> Target<'static> {
        Target {
            bbox: [28.0 / 64.0, 20.0 / 64.0, 36.0 / 64.0, 44.0 / 64.0],
            silhouette: None,
            shift: (0.0, 0.0),
        }
    }

    #[test]
    fn erasing_a_player_leaves_grass() {
        let mut frame = grass_with_player();
        erase(&mut frame, &target(), &Mapping::IDENTITY, 1.0);
        let middle = frame.pixel(32, 32).unwrap();
        assert!(middle[1] > middle[0] + 50, "{middle:?}");
    }

    #[test]
    fn isolating_a_player_clears_the_grass() {
        let mut frame = grass_with_player();
        isolate(&mut frame, &target(), &Mapping::IDENTITY);
        assert_eq!(frame.pixel(2, 2).unwrap()[3], 0);
        assert_eq!(frame.pixel(32, 32).unwrap()[3], 255);
    }

    #[test]
    fn a_spotlight_darkens_the_far_corner_only() {
        let mut frame = grass_with_player();
        spotlight(
            &mut frame,
            &[(target(), Shape::DEFAULT)],
            &Mapping::IDENTITY,
            1.0,
            1.0,
        );
        assert!(frame.pixel(1, 1).unwrap()[1] < 100);
        assert_eq!(frame.pixel(32, 32).unwrap(), [230, 30, 30, 255]);
    }

    #[test]
    fn shading_tints_the_player() {
        let mut frame = grass_with_player();
        shade(&mut frame, &target(), &Mapping::IDENTITY, [0, 0, 255], 1.0);
        assert!(frame.pixel(32, 32).unwrap()[2] > 100);
        assert_eq!(frame.pixel(2, 2).unwrap(), [40, 140, 50, 255]);
    }

    #[test]
    fn a_bigger_ring_reaches_further() {
        let mut small = grass_with_player();
        let mut big = grass_with_player();
        let shape = Shape::DEFAULT;
        ring(
            &mut small,
            &target(),
            &Mapping::IDENTITY,
            1.0,
            [0, 0, 255],
            1.0,
            shape,
        );
        let bigger = Shape { size: 2.0, ..shape };
        ring(
            &mut big,
            &target(),
            &Mapping::IDENTITY,
            1.0,
            [0, 0, 255],
            1.0,
            bigger,
        );
        let blue = |frame: &Frame| {
            (0..64)
                .filter(|&x| frame.pixel(x, 43).unwrap()[2] > 120)
                .count()
        };
        assert!(blue(&big) > blue(&small), "{} {}", blue(&big), blue(&small));
    }

    #[test]
    fn an_arrow_points_down_at_the_player() {
        let mut frame = grass_with_player();
        arrow(
            &mut frame,
            &target(),
            &Mapping::IDENTITY,
            1.0,
            [0, 0, 255],
            1.0,
            Shape::DEFAULT,
        );
        // Above the head, in the middle, it is blue; under the feet it is not.
        assert!(
            frame.pixel(32, 10).unwrap()[2] > 120,
            "{:?}",
            frame.pixel(32, 10)
        );
        assert_eq!(frame.pixel(32, 50).unwrap(), [40, 140, 50, 255]);
    }

    #[test]
    fn a_frame_rings_the_box_and_leaves_the_middle() {
        let mut frame = grass_with_player();
        frame_round(
            &mut frame,
            &target(),
            &Mapping::IDENTITY,
            [0, 0, 255],
            1.0,
            Shape::DEFAULT,
        );
        assert_eq!(frame.pixel(32, 32).unwrap(), [230, 30, 30, 255]);
        let blue_edges = (0..64)
            .filter(|&y| frame.pixel(27, y).unwrap()[2] == 255)
            .count();
        assert!(blue_edges > 10, "{blue_edges}");
    }

    #[test]
    fn a_wider_frame_reaches_further_across_only() {
        let draw = |shape: Shape| {
            let mut frame = grass_with_player();
            frame_round(
                &mut frame,
                &target(),
                &Mapping::IDENTITY,
                [0, 0, 255],
                1.0,
                shape,
            );
            frame
        };
        let blue = |frame: &Frame, x, y| frame.pixel(x, y).unwrap()[2] == 255;
        let left = |frame: &Frame| (0..64).find(|&x| blue(frame, x, 32)).unwrap();
        let top = |frame: &Frame| (0..64).find(|&y| blue(frame, 32, y)).unwrap();
        let normal = draw(Shape::DEFAULT);
        let wide = draw(Shape {
            width: 2.0,
            ..Shape::DEFAULT
        });
        assert!(
            left(&wide) < left(&normal),
            "{} {}",
            left(&wide),
            left(&normal)
        );
        assert_eq!(top(&wide), top(&normal));
    }

    #[test]
    fn pull_push_fills_a_hole_with_its_surroundings() {
        let colour = vec![[100.0, 100.0, 100.0]; 16];
        let mut known = vec![1.0; 16];
        known[5] = 0.0;
        known[6] = 0.0;
        let mut holey = colour.clone();
        holey[5] = [0.0; 3];
        holey[6] = [0.0; 3];
        let filled = pull_push(&holey, &known, 4, 4);
        assert!((filled[5][0] - 100.0).abs() < 1.0);
    }
}
