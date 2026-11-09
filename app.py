import html
import gradio as gr
import inference_2 as inference
import numpy as np
import uvicorn

from auth_store import AuthStore
from auth_web import create_auth_app
from frame_analysis import FrameAnalysisError, extract_sampled_video_frames
from grad_cam import FeatureAttribution


def _render_attribution_cards(
    results: list[tuple[str, tuple[FeatureAttribution, ...], str]],
    empty_message: str = "No face regions to explain. Authenticity was not assessed.",
) -> str:
    if not results:
        return (
            f'<p class="attribution-empty">{html.escape(empty_message)}</p>'
        )

    sections = []
    for heading, features, explanation_class in results:
        face_groups: dict[str, list[tuple[FeatureAttribution, str]]] = {}
        for feature in features:
            face_label, separator, feature_name = feature.name.partition(" · ")
            if not separator or not face_label.startswith("Face "):
                face_label, feature_name = "Face regions", feature.name
            face_groups.setdefault(face_label, []).append((feature, feature_name))

        for face_label, face_features in face_groups.items():
            cards = []
            for feature, feature_name in sorted(
                face_features,
                key=lambda item: item[0].rank,
            ):
                percentage = max(0.0, min(1.0, feature.score)) * 100
                cards.append(
                    f"""
                    <article class="attribution-card">
                        <div class="attribution-card-top">
                            <span class="attribution-rank">#{feature.rank}</span>
                            <span class="attribution-name">{html.escape(feature_name)}</span>
                            <strong>{percentage:.0f}%</strong>
                        </div>
                        <div class="attribution-track">
                            <span style="width:{percentage:.1f}%"></span>
                        </div>
                        <small>Relative Grad-CAM attention · approximate region</small>
                    </article>
                    """
                )
            sections.append(
                f"""
                <section class="attribution-result">
                    <div class="attribution-heading">
                        <strong>{html.escape(heading)} · {html.escape(face_label)}</strong>
                        <span>Explains the model's {html.escape(explanation_class)} class output</span>
                    </div>
                    <div class="attribution-grid">{''.join(cards)}</div>
                </section>
                """
            )

    return (
        '<div class="attribution-panel">'
        '<div class="attribution-disclaimer">Grad-CAM ranks where the model was '
        "sensitive, not whether a feature is fake or why an error occurred.</div>"
        f"{''.join(sections)}</div>"
    )


def analyze_image_outputs(
    image_rgb: np.ndarray | None,
) -> tuple[str, np.ndarray | None, str]:
    prediction = inference.analyze_image(image_rgb, with_explanation=True)
    attribution_html = _render_attribution_cards(
        [
            (
                f"Image analysis · {prediction.verdict}",
                prediction.feature_attributions,
                prediction.explanation_class or "selected",
            )
        ]
        if prediction.feature_attributions
        else [],
        empty_message=(
            "Choose an image to see its analysis."
            if prediction.verdict == "NO_IMAGE"
            else "No face detected. Authenticity was not assessed."
        ),
    )
    return (
        prediction.summary,
        prediction.explanation_image_rgb,
        attribution_html,
    )


def analyze_video_frames(
    video_path: str | None,
    sample_count: int,
) -> tuple[
    str,
    list[tuple[np.ndarray, str]],
    list[list[str]],
    list[tuple[np.ndarray, str]],
    str,
]:
    try:
        _, frames = extract_sampled_video_frames(video_path, sample_count)
    except FrameAnalysisError as exc:
        return str(exc), [], [], [], ""

    gallery_items = []
    ranked_items = []
    predictions = []
    model_predictions = []
    attribution_results = []
    for frame in frames:
        timestamp = (
            f"{frame.timestamp_seconds:.2f} s"
            if frame.timestamp_seconds is not None
            else "timestamp unavailable"
        )
        prediction = inference.analyze_image(
            frame.image_rgb,
            with_explanation=True,
        )
        model_predictions.append(prediction)
        fake_score = (
            f"{prediction.fake_score:.1%}"
            if prediction.fake_score is not None
            else "Not available"
        )
        real_score = (
            f"{prediction.real_score:.1%}"
            if prediction.real_score is not None
            else "Not available"
        )
        gallery_items.append(
            (
                frame.image_rgb,
                f"Frame {frame.frame_number} · {timestamp} · {prediction.verdict}",
            )
        )
        if prediction.fake_score is not None:
            ranked_items.append(
                (
                    prediction.fake_score,
                    (
                        prediction.explanation_image_rgb
                        if prediction.explanation_image_rgb is not None
                        else frame.image_rgb
                    ),
                    f"Frame {frame.frame_number} · {timestamp} · "
                    f"fake model score {fake_score} · "
                    f"explains {prediction.explanation_class or 'selected'} output",
                )
            )
            attribution_results.append(
                (
                    f"Frame {frame.frame_number} · {timestamp} · {prediction.verdict}",
                    prediction.feature_attributions,
                    prediction.explanation_class or "selected",
                )
            )
        predictions.append(
            [
                str(frame.frame_number),
                timestamp,
                str(prediction.face_count),
                prediction.verdict,
                real_score,
                fake_score,
                "; ".join(
                    f"#{feature.rank} {feature.name}: {feature.score:.0%}"
                    for feature in sorted(
                        prediction.feature_attributions,
                        key=lambda item: item.rank,
                    )
                ),
            ]
        )

    ranked_items.sort(key=lambda item: item[0], reverse=True)
    status = inference.summarize_video_predictions(model_predictions, len(frames))
    return (
        status,
        gallery_items,
        predictions,
        [(image, caption) for _, image, caption in ranked_items],
        _render_attribution_cards(attribution_results),
    )


# ============================================================
# CUSTOM CSS
# ============================================================

CSS = """

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap');


/* ============================================================
   COLORS
   ============================================================ */

:root {
    --cream: #f4f1e5;
    --cream-light: #faf8ef;
    --dark: #18251d;
    --olive: #777451;
    --border: #d8d5c5;
    --muted: #65685e;
}
.console-heading span:first-child { color: #f0f1f5 !important; }


/* ============================================================
   GLOBAL
   ============================================================ */

html,
body {
    margin: 0 !important;
    padding: 0 !important;

    background: var(--cream) !important;

    font-family: "DM Sans", sans-serif !important;
    color: var(--dark) !important;
}


/* Main Gradio container */

.gradio-container {
    width: 100% !important;
    max-width: 1600px !important;

    margin: 0 auto !important;
    padding: 0 45px 60px !important;

    background: var(--cream) !important;
}


/* Remove unnecessary default spacing */

.gradio-container > .main {
    width: 100% !important;
}


/* ============================================================
   HEADER / NAVBAR
   ============================================================ */

.header {
    width: 100%;

    padding: 28px 0;

    border-bottom: 1px solid var(--border);

    display: flex;
    justify-content: space-between;
    align-items: center;
}


.logo {
    font-family: "Playfair Display", serif;

    font-size: 32px;
    font-weight: 600;

    letter-spacing: -1px;

    color: var(--dark);
}


.logo span {
    color: var(--olive);
    font-style: italic;
}


.nav {
    display: flex;

    gap: 38px;

    font-size: 15px;
    font-weight: 500;

    color: #55584f;
}


.nav span {
    cursor: pointer;

    transition: color 0.2s ease;
}


.nav span:hover {
    color: var(--olive);
}

.nav a {
    color: inherit;
    text-decoration: none;
}

.nav a:hover {
    color: var(--olive);
}

.logout-form {
    display: inline-flex;
    margin: 0;
}

.logout-form button {
    padding: 0;
    border: 0;
    background: transparent;
    color: inherit;
    cursor: pointer;
    font: inherit;
}

.logout-form button:hover {
    color: var(--olive);
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    width: 100%;

    text-align: center;

    padding: 105px 20px 85px;

    box-sizing: border-box;
}


.hero h1 {
    max-width: 1100px;

    margin: 0 auto;

    font-family: "Playfair Display", serif;

    font-size: clamp(55px, 6.5vw, 95px);

    line-height: 0.98;

    letter-spacing: -4px;

    font-weight: 500;

    color: var(--dark);
}


.hero h1 span {
    color: var(--olive);

    font-style: italic;
}


.hero p {
    max-width: 760px;

    margin: 32px auto 0;

    font-size: 18px;

    line-height: 1.75;

    color: var(--muted);
}


/* ============================================================
   MAIN DETECTION CARD
   ============================================================ */

.main {
    width: 100% !important;

    box-sizing: border-box;

    background: var(--cream-light);

    border: 1px solid var(--border);

    border-radius: 28px;

    margin: 0 auto 55px !important;

    padding: 42px !important;
}


/* Section title */

.section-title {
    font-family: "Playfair Display", serif !important;

    font-size: 34px !important;

    color: var(--dark) !important;

    margin-bottom: 5px !important;
}


/* Section subtitle */

.section-text {
    color: var(--muted) !important;

    margin-bottom: 28px;
}


/* ============================================================
   TABS
   ============================================================ */

.tab-nav {
    border-bottom: 1px solid var(--border) !important;

    background: transparent !important;
}


.tab-nav button {
    background: transparent !important;

    color: #6b6d64 !important;

    font-family: "DM Sans", sans-serif !important;

    font-weight: 600 !important;

    border: none !important;

    padding: 14px 20px !important;
}


.tab-nav button:hover {
    color: var(--dark) !important;
}


.tab-nav button.selected {
    color: var(--dark) !important;

    border-bottom: 2px solid var(--dark) !important;
}


/* ============================================================
   UPLOAD AREA
   ============================================================ */

.upload-area {
    width: 100% !important;

    min-height: 300px !important;

    border: 2px dashed #aaa995 !important;

    border-radius: 22px !important;

    background: #f7f5eb !important;

    overflow: hidden !important;

    margin-top: 20px !important;
}


.upload-area:hover {
    border-color: var(--olive) !important;
}


/* ============================================================
   ANALYZE BUTTON
   ============================================================ */

.analyze {
    width: 100% !important;

    background: var(--dark) !important;

    color: white !important;

    border: none !important;

    border-radius: 999px !important;

    min-height: 55px !important;

    font-family: "DM Sans", sans-serif !important;

    font-size: 16px !important;

    font-weight: 600 !important;

    margin-top: 18px !important;

    transition: all 0.2s ease !important;
}


.analyze:hover {
    background: #304737 !important;

    transform: translateY(-1px);
}


/* ============================================================
   RESULT
   ============================================================ */

.result textarea {
    background: var(--dark) !important;

    color: white !important;

    border: none !important;

    border-radius: 18px !important;

    font-size: 17px !important;

    line-height: 1.5 !important;
}


.result label {
    color: var(--dark) !important;

    font-weight: 600 !important;
}


/* ============================================================
   INFORMATION SECTION
   ============================================================ */

.info {
    width: 100%;

    box-sizing: border-box;

    padding: 10px 0 70px;

    display: grid;

    grid-template-columns: repeat(3, 1fr);

    gap: 22px;
}


.info-card {
    background: var(--cream-light);

    border: 1px solid var(--border);

    border-radius: 22px;

    padding: 30px;

    min-height: 180px;

    box-sizing: border-box;
}


.number {
    font-family: "Playfair Display", serif;

    font-size: 40px;

    color: var(--olive);
}


.info-card h3 {
    font-family: "Playfair Display", serif;

    font-size: 26px;

    color: var(--dark);

    margin: 10px 0;
}


.info-card p {
    color: var(--muted);

    line-height: 1.65;

    margin: 0;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    width: 100%;

    box-sizing: border-box;

    background: var(--dark);

    color: white;

    text-align: center;

    padding: 50px 25px;

    border-radius: 22px;
}


.footer h2 {
    font-family: "Playfair Display", serif;

    font-size: 34px;

    margin: 0 0 10px;
}


.footer p {
    color: #c2c8be;

    margin: 0;
}


/* ============================================================
   GENERAL INPUTS
   ============================================================ */

label {
    color: var(--dark) !important;
}


input,
textarea {
    font-family: "DM Sans", sans-serif !important;
}


/* ============================================================
   LARGE DESKTOP
   ============================================================ */

@media (min-width: 1400px) {

    .gradio-container {
        padding-left: 70px !important;
        padding-right: 70px !important;
    }

    .hero {
        padding-top: 120px;
        padding-bottom: 100px;
    }

    .hero h1 {
        font-size: 96px;
    }

}


/* ============================================================
   TABLET
   ============================================================ */

@media (max-width: 1000px) {

    .gradio-container {
        padding-left: 28px !important;
        padding-right: 28px !important;
    }


    .hero {
        padding: 80px 15px 65px;
    }


    .hero h1 {
        font-size: clamp(50px, 8vw, 75px);

        letter-spacing: -3px;
    }


    .main {
        padding: 30px !important;
    }


    .info {
        grid-template-columns: 1fr 1fr;
    }

}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 700px) {

    .gradio-container {
        width: 100% !important;

        max-width: 100% !important;

        padding: 0 16px 35px !important;
    }


    /* HEADER */

    .header {
        padding: 20px 0;

        flex-direction: row;
    }


    .logo {
        font-size: 26px;
    }


    .nav {
        gap: 15px;

        font-size: 12px;
    }


    /* HERO */

    .hero {
        padding: 60px 5px 50px;
    }


    .hero h1 {
        font-size: 45px;

        line-height: 1.03;

        letter-spacing: -2px;
    }


    .hero p {
        font-size: 15px;

        line-height: 1.65;

        margin-top: 24px;
    }


    /* MAIN */

    .main {
        margin: 0 auto 35px !important;

        padding: 22px 16px !important;

        border-radius: 20px !important;
    }


    .section-title {
        font-size: 28px !important;
    }


    .section-text {
        font-size: 14px;

        line-height: 1.5;
    }


    /* TABS */

    .tab-nav {
        overflow-x: auto !important;

        white-space: nowrap !important;
    }


    .tab-nav button {
        padding: 10px 12px !important;

        font-size: 13px !important;
    }


    /* UPLOAD */

    .upload-area {
        min-height: 240px !important;

        margin-top: 15px !important;
    }


    /* BUTTON */

    .analyze {
        min-height: 50px !important;

        font-size: 15px !important;
    }


    /* INFO */

        grid-template-columns: 1fr;

        gap: 15px;

        padding-bottom: 40px;
    }


    .info-card {
        padding: 24px;

        min-height: auto;
    }


    .info-card h3 {
        font-size: 23px;
    }


    /* FOOTER */

    .footer {
        padding: 40px 20px;

        border-radius: 18px;
    }


    .footer h2 {
        font-size: 29px;
    }

}


/* ============================================================
   SMALL MOBILE
   ============================================================ */

@media (max-width: 420px) {

    .gradio-container {
        padding-left: 12px !important;
        padding-right: 12px !important;
    }


    .logo {
        font-size: 23px;
    }


    .nav {
        gap: 10px;

        font-size: 11px;
    }


    .hero {
        padding-top: 48px;
    }


    .hero h1 {
        font-size: 38px;

        letter-spacing: -1.5px;
    }


    .hero p {
        font-size: 14px;
    }


    .main {
        padding: 18px 13px !important;
    }


    .section-title {
        font-size: 25px !important;
    }


    .tab-nav button {
        padding: 8px 9px !important;

        font-size: 12px !important;
    }

}

"""

CSS += """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --page: #f5f6f8;
    --paper: #ffffff;
    --ink: #17181d;
    --muted: #686d78;
    --line: #e2e4e9;
    --accent: #4158f5;
    --accent-dark: #3048df;
    --hero: #101116;
}

html, body {
    background: var(--page) !important;
    color: var(--ink) !important;
    font-family: "DM Sans", "Segoe UI", sans-serif !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 0 clamp(16px, 5vw, 72px) 60px !important;
    background: var(--page) !important;
}

body { overflow-x: hidden !important; }
.main.fillable.app {
    max-width: none !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: var(--page) !important;
    box-shadow: none !important;
}

.header {
    position: relative;
    z-index: 3;
    width: 100vw;
    margin-left: calc(50% - 50vw);
    margin-bottom: -24px !important;
    padding: 0 clamp(16px, 5vw, 72px);
    border: 0;
    background: var(--hero);
    color: #fff;
}

.header-inner {
    width: min(100%, 1296px);
    min-height: 76px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 28px;
}

.logo {
    flex: 0 0 auto;
    color: #fff;
    font-family: "DM Sans", "Segoe UI", sans-serif;
    font-size: 23px;
    font-weight: 700;
    letter-spacing: -0.8px;
    text-decoration: none;
}

.logo span { color: #9eabff; font-style: normal; }

.nav {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: clamp(16px, 2.4vw, 34px);
    color: #c5c7d0;
    font-size: 14px;
    font-weight: 500;
    white-space: nowrap;
}

.nav a {
    color: inherit;
    text-decoration: none;
    transition: color 0.18s ease;
}

.nav a:hover, .nav a:focus-visible { color: #fff; }

.account-controls {
    display: flex;
    align-items: center;
    gap: 14px;
    color: #b7bac5 !important;
    font-size: 13px;
    white-space: nowrap;
}
.account-controls .account-label { color: #b7bac5 !important; }

.logout-form { display: inline-flex; margin: 0; }
.logout-form button {
    width: auto;
    min-height: 40px;
    margin: 0;
    padding: 0 17px;
    border: 1px solid #464852;
    border-radius: 8px;
    background: #fff;
    color: #17181d;
    cursor: pointer;
    font: inherit;
    font-weight: 650;
}
.logout-form button:hover { background: #e9ebf2; color: #17181d; }

.hero {
    width: 100vw;
    margin-left: calc(50% - 50vw);
    padding: clamp(50px, 7vw, 92px) clamp(16px, 5vw, 72px);
    background: var(--hero);
    color: #fff;
    text-align: left;
}

.hero-inner {
    width: min(100%, 1296px);
    margin: 0 auto;
    display: grid;
    grid-template-columns: minmax(0, 1.05fr) minmax(300px, 0.75fr);
    align-items: center;
    gap: clamp(35px, 7vw, 110px);
}

.hero-copy { max-width: 690px; }
.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    margin: 0 0 22px;
    color: #a7adbf;
    font-size: 12px;
    font-weight: 650;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.eyebrow::before {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #7285ff;
    content: "";
    box-shadow: 0 0 0 5px rgb(65 88 245 / 16%);
}

.hero h1 {
    max-width: 760px;
    margin: 0;
    color: #f7f7fa;
    font-family: "DM Sans", "Segoe UI", sans-serif;
    font-size: clamp(44px, 6vw, 76px);
    font-weight: 650;
    letter-spacing: -0.065em;
    line-height: 1.02;
}

.hero h1 span { color: #a9b4ff; font-style: normal; }
.hero p {
    max-width: 590px;
    margin: 24px 0 0;
    color: #b6b9c4;
    font-size: clamp(15px, 1.4vw, 18px);
    line-height: 1.7;
}

.hero-actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 17px;
    margin-top: 30px;
}
.hero-action {
    display: inline-flex;
    min-height: 48px;
    align-items: center;
    justify-content: center;
    padding: 0 20px;
    border-radius: 8px;
    background: var(--accent);
    color: #fff;
    font-size: 14px;
    font-weight: 650;
    text-decoration: none;
    transition: background 0.18s ease, transform 0.18s ease;
}
.hero-action:hover { transform: translateY(-1px); background: var(--accent-dark); }
.hero-secondary { color: #d3d5dd; font-size: 14px; text-decoration: none; }
.hero-secondary:hover { color: #fff; }

.hero-console {
    padding: 23px;
    border: 1px solid #30323b;
    border-radius: 14px;
    background: #191b22;
    box-shadow: 0 28px 70px rgb(0 0 0 / 22%);
}
.console-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 17px;
    border-bottom: 1px solid #30323b;
    color: #e9eaf0 !important;
    font-size: 13px;
    font-weight: 650;
}
.console-heading span:last-child {
    color: #9298ab;
    font-size: 11px;
    font-weight: 500;
}
.console-list { display: grid; gap: 12px; margin-top: 17px; }
.console-row {
    display: grid;
    grid-template-columns: 39px 1fr;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border: 1px solid #2e3039;
    border-radius: 10px;
    background: #20222a;
}
.console-icon {
    display: grid;
    width: 38px;
    height: 38px;
    place-items: center;
    border-radius: 9px;
    background: #292e50;
    color: #aab4ff;
    font-size: 15px;
    font-weight: 700;
}
.console-row strong, .console-row small { display: block; }
.console-row strong { color: #f0f1f5; font-size: 13px; }
.console-row small { margin-top: 4px; color: #a4a8b5; font-size: 11px; line-height: 1.45; }
.console-footnote { margin: 14px 0 0 !important; color: #8c909d !important; font-size: 11px !important; }

#detection.main {
    width: min(100%, 1120px) !important;
    box-sizing: border-box;
    margin: 62px auto 72px !important;
    padding: clamp(22px, 4vw, 42px) !important;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: var(--paper);
    box-shadow: 0 16px 48px rgb(21 24 33 / 5%);
}

.section-title {
    margin: 0 0 6px !important;
    color: var(--ink) !important;
    font-family: "DM Sans", "Segoe UI", sans-serif !important;
    font-size: clamp(25px, 3vw, 32px) !important;
    font-weight: 650 !important;
    letter-spacing: -0.04em;
}
.section-text { margin: 0 0 24px; color: var(--muted) !important; line-height: 1.65; }

.tab-container {
    gap: 5px;
    border-bottom: 1px solid var(--line) !important;
    background: transparent !important;
}
.tab-container [role="tab"] {
    padding: 14px 17px !important;
    border: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    color: #737783 !important;
    font-family: "DM Sans", "Segoe UI", sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}
.tab-container [role="tab"]:hover { color: var(--ink) !important; }
.tab-container [role="tab"][aria-selected="true"] {
    border-bottom-color: var(--accent) !important;
    color: var(--ink) !important;
}
.tab-nav {
    gap: 5px;
    border-bottom: 1px solid var(--line) !important;
    background: transparent !important;
}
.tab-nav button {
    padding: 14px 17px !important;
    border: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    color: #737783 !important;
    font-family: "DM Sans", "Segoe UI", sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}
.tab-nav button:hover { color: var(--ink) !important; }
.tab-nav button.selected {
    border-bottom-color: var(--accent) !important;
    color: var(--ink) !important;
}

.upload-instructions { margin: 20px 0 4px !important; color: var(--muted) !important; font-size: 14px; }
.upload-area {
    width: 100% !important;
    min-height: 260px !important;
    margin-top: 12px !important;
    overflow: hidden !important;
    border: 1px dashed #c9cdd7 !important;
    border-radius: 12px !important;
    background: #fafbfc !important;
    transition: border-color 0.18s ease, background 0.18s ease;
}
.upload-area:hover { border-color: var(--accent) !important; background: #f8f9ff !important; }

.analyze {
    width: 100% !important;
    min-height: 50px !important;
    margin-top: 18px !important;
    border: 0 !important;
    border-radius: 8px !important;
    background: var(--accent) !important;
    color: #fff !important;
    font-family: "DM Sans", "Segoe UI", sans-serif !important;
    font-size: 15px !important;
    font-weight: 650 !important;
    transition: background 0.18s ease, transform 0.18s ease;
}
.analyze:hover { transform: translateY(-1px); background: var(--accent-dark) !important; }
.analyze:disabled { cursor: not-allowed; opacity: 0.6; }

.result { margin-top: 20px; }
.result textarea {
    min-height: 100px;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    background: #f7f8fa !important;
    color: var(--ink) !important;
    font-family: "DM Sans", "Segoe UI", sans-serif !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}
.result label { color: var(--ink) !important; font-weight: 650 !important; }
.result-note { margin: 8px 0 0 !important; color: var(--muted) !important; font-size: 12px; line-height: 1.55; }
.frame-analysis {
    margin-top: 30px;
    padding-top: 25px;
    border-top: 1px solid var(--line);
}
.frame-analysis-title { margin: 0 0 6px; color: var(--ink); font-size: 20px; font-weight: 650; }
.frame-analysis-copy { margin: 0 0 16px; color: var(--muted); font-size: 13px; line-height: 1.6; }
.frame-options { align-items: end; }
.frame-options .analyze { margin-top: 0 !important; }
.frame-gallery img { border-radius: 9px; }

.content-section { width: min(100%, 1120px); margin: 0 auto 72px; }
.content-heading { margin: 0 0 10px; color: var(--ink); font-size: clamp(25px, 3vw, 34px); font-weight: 650; letter-spacing: -0.045em; }
.content-lead { max-width: 720px; margin: 0 0 25px; color: var(--muted); line-height: 1.7; }
.info {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    width: 100%;
    padding: 4px 0 0;
}
.info-card {
    min-height: 185px;
    box-sizing: border-box;
    padding: 23px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #fff;
}
.number { color: var(--accent); font-size: 12px; font-weight: 700; letter-spacing: 0.12em; }
.info-card h3 { margin: 19px 0 9px; color: var(--ink); font-size: 19px; font-weight: 650; }
.info-card p { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.65; }

.details-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.detail-card { padding: 22px; border: 1px solid var(--line); border-radius: 12px; background: #fff; }
.detail-card h3 { margin: 0 0 9px; color: var(--ink); font-size: 16px; }
.detail-card p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.65; }

.example-panel {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 22px;
    padding: 24px;
    border: 1px solid #dfe2eb;
    border-radius: 12px;
    background: #fff;
}
.example-panel strong { display: block; margin-bottom: 5px; color: var(--ink); }
.example-panel p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.6; }
.example-label { padding: 8px 11px; border-radius: 999px; background: #f0f2ff; color: #3048df; font-size: 11px; font-weight: 700; white-space: nowrap; }

.footer {
    width: 100vw;
    box-sizing: border-box;
    margin-left: calc(50% - 50vw);
    padding: 34px clamp(16px, 5vw, 72px);
    border-radius: 0;
    background: var(--hero);
    color: #fff;
    text-align: left;
}
.footer-inner { width: min(100%, 1296px); margin: 0 auto; display: flex; justify-content: space-between; gap: 24px; align-items: center; }
.footer h2 { margin: 0 0 5px; color: #fff; font-family: "DM Sans", "Segoe UI", sans-serif; font-size: 20px; }
.footer p { margin: 0; color: #aeb1bd; font-size: 12px; line-height: 1.6; }
.footer-links { display: flex; gap: 20px; flex-wrap: wrap; }
.footer-links a { color: #c9cbd3; font-size: 12px; text-decoration: none; }
.footer-links a:hover { color: #fff; }

label { color: var(--ink) !important; }
input, textarea { font-family: "DM Sans", "Segoe UI", sans-serif !important; }
:focus-visible { outline: 3px solid rgb(65 88 245 / 70%); outline-offset: 3px; }
.gradio-container footer:not(.footer) { display: none !important; }

@media (max-width: 900px) {
    .header-inner { flex-wrap: wrap; gap: 12px 20px; padding: 15px 0; }
    .nav { order: 3; width: 100%; justify-content: flex-start; overflow-x: auto; padding: 4px 0; }
    .hero-inner { grid-template-columns: 1fr; gap: 34px; }
    .hero-console { max-width: 620px; }
    .info, .details-grid { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 600px) {
    .gradio-container { padding: 0 16px 36px !important; }
    .header-inner { align-items: center; }
    .logo { font-size: 21px; }
    .account-label { display: none; }
    .logout-form button { min-height: 37px; padding: 0 12px; }
    .nav {
        gap: 8px 18px;
        overflow: visible;
        white-space: normal;
        flex-wrap: wrap;
        font-size: 13px;
    }
    .hero { padding-top: 46px; padding-bottom: 47px; }
    .hero h1 { font-size: clamp(39px, 12vw, 56px); letter-spacing: -0.06em; }
    .hero p { margin-top: 17px; font-size: 15px; }
    .hero-console { padding: 16px; }
    #detection.main { margin: 36px auto 48px !important; padding: 19px 16px !important; border-radius: 13px; }
    .tab-container { overflow-x: auto !important; white-space: nowrap !important; }
    .tab-container [role="tab"] { padding: 12px 11px !important; font-size: 12px !important; }
    .tab-nav { overflow-x: auto !important; white-space: nowrap !important; }
    .tab-nav button { padding: 12px 11px !important; font-size: 12px !important; }
    .upload-area { min-height: 210px !important; }
    .info, .details-grid { grid-template-columns: 1fr; }
    .content-section { margin-bottom: 50px; }
    .example-panel { grid-template-columns: 1fr; }
    .example-label { justify-self: start; }
    .footer-inner { align-items: flex-start; flex-direction: column; }
}

.attribution-panel { display: grid; gap: 18px; margin: 14px 0 22px; }
.attribution-disclaimer, .attribution-empty {
    margin: 0;
    padding: 13px 15px;
    border: 1px solid #dfe3f3;
    border-radius: 10px;
    background: #f5f6ff;
    color: #555d78;
    font-size: 12px;
    line-height: 1.6;
}
.attribution-result {
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 13px;
    background: #fff;
    box-shadow: 0 8px 26px rgb(21 24 33 / 4%);
}
.attribution-heading {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 6px 16px;
    padding: 15px 17px;
    border-bottom: 1px solid var(--line);
}
.attribution-heading strong { color: var(--ink); font-size: 14px; }
.attribution-heading span { color: var(--muted); font-size: 11px; }
.attribution-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 10px;
    padding: 13px;
}
.attribution-card {
    min-width: 0;
    padding: 13px;
    border: 1px solid #e8eaf0;
    border-radius: 10px;
    background: linear-gradient(145deg, #fff, #fafbff);
}
.attribution-card-top { display: flex; align-items: center; gap: 8px; }
.attribution-rank {
    display: inline-grid;
    width: 27px;
    height: 27px;
    flex: 0 0 auto;
    place-items: center;
    border-radius: 8px;
    background: #edf0ff;
    color: #394fd1;
    font-size: 11px;
    font-weight: 750;
}
.attribution-name { flex: 1; color: var(--ink); font-size: 12px; font-weight: 650; }
.attribution-card-top strong { color: #3349cf; font-size: 14px; }
.attribution-track { height: 5px; overflow: hidden; margin: 11px 0 8px; border-radius: 9px; background: #eceef5; }
.attribution-track span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #7182ff, #4258ee); }
.attribution-card small { color: var(--muted); font-size: 10px; }
.attribution-output { margin-top: 14px; }
.frame-gallery img { border-radius: 10px; }

@media (max-width: 600px) {
    .attribution-grid { grid-template-columns: 1fr; }
    .attribution-heading { flex-direction: column; }
}
"""


# ============================================================
# HEADER
# ============================================================

HEADER = """
<header class="header">
    <div class="header-inner">
        <a class="logo" href="#home" aria-label="DeepTrace home">Deep<span>Trace</span></a>
        <nav class="nav" aria-label="Primary navigation">
            <a href="#home">Home</a>
            <a href="#detection" data-detector-tab="Video Detection">Video Detector</a>
            <a href="#detection" data-detector-tab="Image Detection">Image Detector</a>
            <a href="#how-it-works">How It Works</a>
            <a href="#history">History</a>
        </nav>
        <div class="account-controls">
            <span class="account-label">Local workspace</span>
            <form class="logout-form" action="/logout" method="post">
                <button type="submit">Sign out</button>
            </form>
        </div>
    </div>
</header>
"""


# ============================================================
# HERO
# ============================================================

HERO = """
<section class="hero" id="home">
    <div class="hero-inner">
        <div class="hero-copy">
            <p class="eyebrow">DeepTrace · Local media analysis</p>
            <h1>Look closer at the media you <span>trust.</span></h1>
            <p>
                Explore model-based predictions for video and image uploads.
                Results are estimates from this academic project—not forensic proof.
            </p>
            <div class="hero-actions">
                <a class="hero-action" href="#detection">Start detection <span aria-hidden="true">→</span></a>
                <a class="hero-secondary" href="#how-it-works">See how it works</a>
            </div>
        </div>
        <aside class="hero-console" aria-label="Overview of face-first media analysis">
            <div class="console-heading">
                <span>Analysis workspace</span>
                <span>LOCAL · MODEL-BASED</span>
            </div>
            <div class="console-list">
                <div class="console-row">
                    <span class="console-icon" aria-hidden="true">V</span>
                    <div><strong>Video</strong><small>Face-gated estimate plus sampled frames ranked by fake model score</small></div>
                </div>
                <div class="console-row">
                    <span class="console-icon" aria-hidden="true">I</span>
                    <div><strong>Image</strong><small>Face check before a real/fake model estimate</small></div>
                </div>
            </div>
            <p class="console-footnote">No face means no authenticity score; model scores are not calibrated confidence.</p>
        </aside>
    </div>
</section>
"""


# ============================================================
# INFORMATION CARDS
# ============================================================

INFO = """
<section class="content-section" id="how-it-works">
    <p class="eyebrow">A transparent workflow</p>
    <h2 class="content-heading">From upload to model output</h2>
    <p class="content-lead">
        DeepTrace checks for a face before running its image model and reports
        uncertain scores instead of forcing a binary verdict.
    </p>
    <div class="info">
        <article class="info-card">
            <div class="number">01 · UPLOAD</div>
            <h3>Choose your media</h3>
            <p>Select the video or image tab and upload a file using the Gradio control.</p>
        </article>
        <article class="info-card">
            <div class="number">02 · INFERENCE</div>
            <h3>Check faces first</h3>
            <p>Only frames with detected faces are scored. Grad-CAM overlays and separate, ranked facial-zone cards show approximate model attention—not a manipulation detector.</p>
        </article>
        <article class="info-card">
            <div class="number">03 · RESULT</div>
            <h3>Review the estimate</h3>
            <p>The real or fake label follows the higher model class score. The percentage is the model score, not a calibrated probability.</p>
        </article>
    </div>
</section>
"""

DETAILS = """
<section class="content-section" id="supported-media">
    <p class="eyebrow">Built around the current project</p>
    <h2 class="content-heading">Video and image analysis</h2>
    <p class="content-lead">
        The face model provides an estimate, not forensic proof, and its training
        does not establish support for animated or other out-of-domain media.
    </p>
    <div class="details-grid">
        <article class="detail-card">
            <h3>Video detection</h3>
            <p>The video predictor checks eight evenly spaced frames and reports the highest mean class score from frames where a face is detected.</p>
        </article>
        <article class="detail-card">
            <h3>Frame analysis</h3>
            <p>Choose 3, 5, or 8 representative frames. Each frame is displayed as extracted; frames with detected faces are scored and ranked by fake-class model output.</p>
        </article>
        <article class="detail-card">
            <h3>Image detection</h3>
            <p>The image predictor uses the current ONNX-converted image model and returns its existing prediction text.</p>
        </article>
    </div>
</section>

<section class="content-section" id="result-preview">
    <p class="eyebrow">A note about results</p>
    <h2 class="content-heading">Read attention as attention</h2>
    <div class="example-panel">
        <div>
            <strong>Interpreting model output</strong>
            <p>Grad-CAM explains the selected class output. Boxed eye, mouth, and jawline zones are approximate and ranked by relative heatmap response—not by manipulation probability. Class scores are not calibrated probabilities, and animation detection needs separate training and evaluation.</p>
        </div>
        <span class="example-label">Estimate · not proof</span>
    </div>
</section>

<section class="content-section" id="history">
    <p class="eyebrow">Privacy and current limitations</p>
    <h2 class="content-heading">Detection history is not stored</h2>
    <p class="content-lead">
        This version does not save user-specific analysis history. Keep any result
        you need from the current analysis panel; uploaded media and predictions
        are not added to a history table.
    </p>
</section>
"""


# ============================================================
# FOOTER
# ============================================================

FOOTER = """
<footer class="footer">
    <div class="footer-inner">
        <div>
            <h2>DeepTrace</h2>
            <p>Academic deepfake-detection project. Predictions are probabilistic and may be incorrect.</p>
        </div>
        <nav class="footer-links" aria-label="Footer navigation">
            <a href="#detection">Analyze media</a>
            <a href="#how-it-works">How it works</a>
            <a href="#supported-media">Supported paths</a>
        </nav>
    </div>
</footer>
"""


# ============================================================
# APPLICATION
# ============================================================

with gr.Blocks(
    title="DeepTrace — Deepfake Detection"
) as app:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.HTML(HEADER)


    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    gr.HTML(HERO)


    # --------------------------------------------------------
    # MAIN DETECTION CARD
    # --------------------------------------------------------

    with gr.Column(elem_classes="main", elem_id="detection"):

        gr.Markdown(
            "## Analyze your media",
            elem_classes="section-title"
        )

        gr.Markdown(
            "Choose a media type, upload a file, and run the matching existing "
            "predictor. Predictions are estimates and may be incorrect.",
            elem_classes="section-text"
        )


        # ====================================================
        # MEDIA TABS
        # ====================================================

        with gr.Tabs():


            # =================================================
            # VIDEO DETECTION
            # =================================================

            with gr.Tab("Video Detection"):

                gr.Markdown(
                    "Upload a video to check for faces first, then estimate authenticity "
                    "from evenly sampled frames across the clip (8 by default).",
                    elem_classes="upload-instructions"
                )

                video_input = gr.Video(
                    label="Choose a video",
                    sources=["upload"],
                    elem_classes="upload-area"
                )

                video_button = gr.Button(
                    "Analyze Video",
                    elem_classes="analyze"
                )

                video_output = gr.Textbox(
                    label="Video verdict and analysis",
                    lines=5,
                    interactive=False,
                    elem_classes="result"
                )
                gr.Markdown(
                    "No face means no authenticity verdict. Model scores are not calibrated "
                    "confidence. Animated media may "
                    "be outside the model's training scope. The video analysis includes "
                    "sampled frames, Grad-CAM overlays, and ranked facial-region cards.",
                    elem_classes="result-note"
                )

                with gr.Column(elem_classes="frame-analysis"):
                    gr.Markdown(
                        "### Frame-by-frame evidence",
                        elem_classes="frame-analysis-title"
                    )
                    gr.Markdown(
                        "The Analyze Video action checks for faces, scores each selected "
                        "frame, and displays its Grad-CAM overlay and ranked facial zones.",
                        elem_classes="frame-analysis-copy"
                    )
                    with gr.Row(elem_classes="frame-options"):
                        frame_sample_count = gr.Dropdown(
                            choices=[3, 5, 8],
                            value=8,
                            label="Frames to analyze",
                        )
                    frame_gallery = gr.Gallery(
                        label="Sampled source frames",
                        columns=3,
                        rows=1,
                        height=320,
                        allow_preview=True,
                        elem_classes="frame-gallery",
                    )
                    suspect_gallery = gr.Gallery(
                        label="Grad-CAM overlays ranked by fake model score",
                        columns=3,
                        rows=1,
                        height=320,
                        allow_preview=True,
                        elem_classes="frame-gallery",
                    )
                    frame_attributions = gr.HTML(
                        label="Ranked facial-region attention",
                        elem_classes="attribution-output",
                    )
                    frame_results = gr.Dataframe(
                        headers=[
                            "Frame",
                            "Timestamp",
                            "Faces detected",
                            "Frame result",
                            "Real model score",
                            "Fake model score",
                            "Facial-region ranks · relative Grad-CAM attention",
                        ],
                        datatype=[
                            "str",
                            "str",
                            "str",
                            "str",
                            "str",
                            "str",
                            "str",
                        ],
                        label="Frame-level predictions",
                        interactive=False,
                        wrap=True,
                    )

                    video_button.click(
                        fn=analyze_video_frames,
                        inputs=[video_input, frame_sample_count],
                        outputs=[
                            video_output,
                            frame_gallery,
                            frame_results,
                            suspect_gallery,
                            frame_attributions,
                        ],
                        show_progress="full",
                    )


            # =================================================
            # IMAGE DETECTION
            # =================================================

            with gr.Tab("Image Detection"):

                gr.Markdown(
                    "Upload an image. The detector checks for a face before estimating "
                    "whether the face appears real or manipulated.",
                    elem_classes="upload-instructions"
                )

                image_input = gr.Image(
                    label="Choose an image",
                    sources=["upload"],
                    elem_classes="upload-area"
                )

                image_button = gr.Button(
                    "Analyze Image",
                    elem_classes="analyze"
                )

                image_output = gr.Textbox(
                    label="Model output",
                    lines=4,
                    interactive=False,
                    elem_classes="result",
                )
                image_explanation = gr.Image(
                    label="Grad-CAM overlay · selected class",
                    interactive=False,
                    elem_classes="frame-gallery",
                )
                image_attributions = gr.HTML(elem_classes="attribution-output")
                gr.Markdown(
                    "Images without a detected face are not assigned a real/fake score. "
                    "Displayed model scores are not calibrated probabilities. Grad-CAM "
                    "zones are approximate and indicate model sensitivity, not manipulation.",
                    elem_classes="result-note"
                )

                image_button.click(
                    fn=analyze_image_outputs,
                    inputs=image_input,
                    outputs=[
                        image_output,
                        image_explanation,
                        image_attributions,
                    ],
                )


    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    gr.HTML(INFO)
    gr.HTML(DETAILS)


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.HTML(FOOTER)

    app.load(
        fn=None,
        js="""() => {
            document.querySelectorAll("[data-detector-tab]").forEach((link) => {
                link.addEventListener("click", () => {
                    const label = link.dataset.detectorTab;
                    window.setTimeout(() => {
                        const tab = Array.from(document.querySelectorAll('[role="tab"]'))
                            .find((button) => button.textContent.includes(label)
                                && button.getBoundingClientRect().height > 1);
                        if (tab) {
                            tab.click();
                            return;
                        }
                        const moreTabs = Array.from(document.querySelectorAll("button"))
                            .find((button) => button.getAttribute("aria-label") === "More tabs");
                        if (moreTabs) {
                            moreTabs.click();
                            window.setTimeout(() => {
                                const menuTab = Array.from(document.querySelectorAll("button"))
                                    .find((button) => button.textContent.includes(label)
                                        && button.getBoundingClientRect().height > 1);
                                if (menuTab) menuTab.click();
                            }, 0);
                        }
                    }, 0);
                });
            });
            return [];
        }""",
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    web_app = create_auth_app(
        app,
        store=AuthStore(),
        css=CSS,
        theme=gr.themes.Base(),
    )
    uvicorn.run(web_app, host="127.0.0.1", port=7860)