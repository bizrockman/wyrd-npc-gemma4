"""
Visual style presets for comic strip generation.

Each style defines the artistic direction that gets appended to every
panel prompt. The character and setting descriptions stay constant -
only the rendering style changes.
"""

# ---------------------------------------------------------------------------
# Character & Setting (constant across all styles)
# ---------------------------------------------------------------------------

CHARACTER = {
    "kenji": (
        "Kenji Sato, Japanese male, age 49, lean build, short graying hair "
        "with silver temples, narrow weathered face, deep-set calm eyes, "
        "wearing white cook's uniform with white cotton apron"
    ),
    "tourist": (
        "young foreign tourist, early 20s, backpack, casual clothes, "
        "holding smartphone, curious expression"
    ),
    "regular": (
        "Japanese man, late 40s, worn dark jacket, tired but comfortable, "
        "familiar posture at the counter"
    ),
    "pushy_stranger": (
        "confident man in business casual, open collar shirt, leaning "
        "forward on the counter, assertive body language"
    ),
    "close_friend": (
        "Japanese man, early 50s, relaxed posture, sleeves rolled up, "
        "beer can in hand, easy familiarity"
    ),
}

SETTING = (
    "narrow yokocho alley ramen shop in Shinjuku, Tokyo. "
    "Eight wooden stools at a worn wooden counter. Warm amber lighting. "
    "Steam rising from large simmering pot. Noren curtain at entrance. "
    "Handwritten menu on wall. Tight, intimate space."
)

# ---------------------------------------------------------------------------
# Camera angles per panel type
# ---------------------------------------------------------------------------

CAMERA = {
    "establishing": "wide shot, exterior, looking into the shop from the alley",
    "enter": "medium shot from behind the counter, customer entering",
    "dialogue": "medium close-up, over-the-shoulder or frontal",
    "reaction": "close-up on face, capturing subtle expression",
    "atmosphere": "detail shot, steam, bowls, hands, counter texture",
    "closing": "wide shot, exterior, shop from outside, evening light",
}

# ---------------------------------------------------------------------------
# Art Style Presets
# ---------------------------------------------------------------------------

STYLES = {
    "manga": {
        "name": "Manga",
        "description": "Japanese manga style with screen tones and clean linework",
        "prompt_suffix": (
            "manga style, black and white with screen tones, clean precise "
            "ink linework, high contrast, dramatic lighting, speed lines for "
            "emphasis, Japanese manga panel composition, seinen manga aesthetic, "
            "detailed background art, Taniguchi Jiro inspired"
        ),
        "negative_prompt": (
            "photorealistic, 3D render, western comic, cartoon, chibi, "
            "colorful, watercolor, oil painting, blurry"
        ),
    },
    "franco_belgian": {
        "name": "Franco-Belgian",
        "description": "European bande dessinee style, clean lines, flat colors",
        "prompt_suffix": (
            "Franco-Belgian bande dessinee style, ligne claire, clean outlines, "
            "flat cel-shaded colors, muted warm color palette, detailed "
            "architectural backgrounds, Moebius inspired composition, "
            "European comic book aesthetic, soft shadows"
        ),
        "negative_prompt": (
            "photorealistic, 3D render, anime, manga screen tones, "
            "american superhero comic, sketch, watercolor, blurry"
        ),
    },
    "trigan": {
        "name": "Trigan Empire",
        "description": "Painted illustration style, rich colors, dramatic realism",
        "prompt_suffix": (
            "painted illustration style, rich saturated colors, dramatic "
            "realistic rendering, detailed brushwork, cinematic lighting, "
            "Don Lawrence inspired, gouache painting aesthetic, "
            "lush detailed backgrounds, heroic realism, vintage illustration"
        ),
        "negative_prompt": (
            "photorealistic photo, 3D render, flat colors, manga, anime, "
            "cartoon, sketch, minimalist, line art"
        ),
    },
    "watercolor": {
        "name": "Watercolor",
        "description": "Loose watercolor with ink outlines, atmospheric",
        "prompt_suffix": (
            "watercolor illustration with ink outlines, loose expressive "
            "brushstrokes, warm muted color palette, atmospheric washes, "
            "visible paper texture, editorial illustration style, "
            "delicate linework, Japanese watercolor aesthetic"
        ),
        "negative_prompt": (
            "photorealistic, 3D render, flat digital colors, manga screen "
            "tones, hard edges, vector art, cartoon"
        ),
    },
    "noir": {
        "name": "Noir",
        "description": "High contrast black and white, heavy shadows, noir mood",
        "prompt_suffix": (
            "film noir style, high contrast black and white, heavy shadows, "
            "dramatic chiaroscuro lighting, stark silhouettes, rain-slicked "
            "surfaces, moody atmospheric, Frank Miller Sin City inspired, "
            "ink splatter texture, graphic novel aesthetic"
        ),
        "negative_prompt": (
            "colorful, bright, cheerful, photorealistic, 3D render, "
            "anime, cute, cartoon, watercolor, pastel"
        ),
    },
}

DEFAULT_STYLE = "manga"


def build_panel_prompt(
    scene_description: str,
    characters: list[str],
    camera: str = "dialogue",
    style: str = DEFAULT_STYLE,
) -> dict:
    """
    Build a complete image generation prompt for a single panel.

    Args:
        scene_description: What happens in this panel (from LLM output)
        characters: List of character keys present (e.g. ["kenji", "tourist"])
        camera: Camera angle key from CAMERA dict
        style: Style preset key from STYLES dict

    Returns:
        dict with 'prompt' and 'negative_prompt' ready for the image API
    """
    style_preset = STYLES.get(style, STYLES[DEFAULT_STYLE])
    camera_direction = CAMERA.get(camera, CAMERA["dialogue"])

    char_descriptions = ". ".join(
        CHARACTER[c] for c in characters if c in CHARACTER
    )

    prompt = (
        f"{scene_description}. "
        f"{char_descriptions}. "
        f"Setting: {SETTING} "
        f"Camera: {camera_direction}. "
        f"{style_preset['prompt_suffix']}"
    )

    return {
        "prompt": prompt,
        "negative_prompt": style_preset["negative_prompt"],
    }


def build_full_page_prompt(
    strip: dict,
    style: str = DEFAULT_STYLE,
) -> dict:
    """
    Build a single prompt that generates an entire comic strip page.

    The prompt describes all panels, the title card, and the art style
    in one block so the image model generates a consistent page.
    """
    style_preset = STYLES.get(style, STYLES[DEFAULT_STYLE])

    # Collect all dialogue turns across scenes
    panels = []
    for scene in strip.get("scenes", []):
        scene_id = scene["id"]
        for turn in scene.get("turns", []):
            user_input = turn.get("input", "")
            response = turn.get("response", "")

            # Extract scene action and dialogue from response
            import re
            scene_desc = ""
            matches = re.findall(r'\*\*scene\*\*\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
            if matches:
                scene_desc = matches[0].strip()

            quotes = re.findall(r'"([^"]+)"', response)
            kenji_line = quotes[0] if quotes else ""

            # Clean customer input
            customer_line = user_input.strip().lstrip("*").split("*")[0].strip()
            if customer_line.startswith("*"):
                customer_line = ""

            panels.append({
                "customer_line": customer_line,
                "kenji_line": kenji_line,
                "scene_desc": scene_desc,
                "scene_id": scene_id,
            })

    # Build panel descriptions
    panel_texts = []
    panel_num = 1

    # Title panel
    title = strip.get("title", "Kenji")
    panel_texts.append(
        f'Title panel top-left: "Kenji" in bold hand-lettered style, '
        f'with a small illustration of a stern Japanese ramen cook in '
        f'white apron, standing behind a counter with steam rising.'
    )

    for p in panels:
        if p["customer_line"]:
            panel_texts.append(
                f'Panel {panel_num}: Customer at the counter. '
                f'Speech bubble: "{p["customer_line"]}"'
            )
            panel_num += 1

        action = p["scene_desc"] or "Kenji behind the counter"
        if p["kenji_line"]:
            panel_texts.append(
                f'Panel {panel_num}: {action}. Kenji, Japanese male, 49, '
                f'short graying hair, white cook\'s uniform. '
                f'Speech bubble: "{p["kenji_line"]}"'
            )
        else:
            panel_texts.append(
                f'Panel {panel_num}: {action}. Kenji, Japanese male, 49, '
                f'short graying hair, white cook\'s uniform. No speech.'
            )
        panel_num += 1

    num_panels = panel_num - 1
    panels_block = "\n\n".join(panel_texts)

    prompt = (
        f"A complete {num_panels}-panel comic strip page. "
        f"Reading left to right, top to bottom. "
        f"Setting: {SETTING} "
        f"Consistent character design throughout all panels.\n\n"
        f"{panels_block}\n\n"
        f"{style_preset['prompt_suffix']}"
    )

    return {
        "prompt": prompt,
        "negative_prompt": style_preset.get("negative_prompt", ""),
    }


def list_styles() -> list[dict]:
    """Return available styles with name and description."""
    return [
        {"key": k, "name": v["name"], "description": v["description"]}
        for k, v in STYLES.items()
    ]
