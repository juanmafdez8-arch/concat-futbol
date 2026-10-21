// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! The player analysis for a batch of clips, outside the window: each clip's
//! `tracks.json` lands in the project's football cache, as the app's own
//! "Analizar todo" leaves it. Ignored by default:
//!
//! ```text
//! CONCAT_ANALYSE_PROJECT=<project folder> CONCAT_ANALYSE_VIDEOS="a.mp4;b.mp4" \
//!     cargo test -p concat-host --test football_batch -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use concat_host::cutout::Progress;
use concat_host::{AppDirs, Brushes, FootballRequest, Footballs};
use concat_vision::football::TrackSet;

#[test]
#[ignore]
fn analyses_a_batch() {
    let project = PathBuf::from(std::env::var("CONCAT_ANALYSE_PROJECT").expect("CONCAT_ANALYSE_PROJECT"));
    let videos = std::env::var("CONCAT_ANALYSE_VIDEOS").expect("CONCAT_ANALYSE_VIDEOS");
    let dirs = AppDirs::locate().unwrap();
    let footballs = Footballs::new(&dirs.data, std::sync::Arc::new(Brushes::new(&dirs.data)));
    assert!(footballs.installed(), "no football model in {}", dirs.data.display());

    for video in videos.split(';').filter(|v| !v.is_empty()) {
        let probe = concat_media::probe(video).expect("probe");
        let stream = probe.require_video().expect("a video stream");
        let duration = probe.duration.map(|d| d.as_f64()).unwrap_or(0.0);
        let request = FootballRequest {
            project: project.clone(),
            media_path: video.to_owned(),
            media_size: (stream.width, stream.height),
            ranges: vec![(0.0, duration)],
        };
        let started = std::time::Instant::now();
        let analysed = footballs
            .analyse(&request, None, &mut |_: Progress| {}, &mut || {})
            .expect("analyse");
        let set = TrackSet::load(&request.dir()).expect("tracks written");
        println!(
            "{video}: {analysed} instants in {:.0}s, kits {:?} -> {}",
            started.elapsed().as_secs_f32(),
            set.team_colours,
            request.dir().display()
        );
    }
}
