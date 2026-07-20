"""events_text.py -- single source of event sentences (G1 brief §1.5.f).

Supervisor-owned. Imported by the reliability rig (B) and the experiments (C);
redefined by neither. The harness streams abstract event keys; the LLM needs
sentences.

Gate runs use text_pick="first" (deterministic sentence per key) so LLM
sampling variance is not confounded by sentence-choice variance. The variety is
for the optional P1/P2 realism re-run only.
"""

KEY_TO_TEXT = {
    "nurture": [
        "You feed it and stroke its head until it settles.",
        "You wrap it in a warm blanket and stay beside it while it dozes.",
        "You bring it its favorite food and sit with it while it eats.",
    ],
    "play": [
        "You dangle a toy and chase it around the room together.",
        "You roll a ball across the floor and it pounces after it, and you cheer.",
        "You play hide-and-seek behind the furniture until you are both worn out.",
    ],
    "neutral": [
        "The room is quiet. You are present but do nothing in particular.",
        "You sit nearby, reading, without looking up at it.",
        "You move about the room doing your own chores, saying nothing.",
    ],
    "neglect": [
        "You walk past without a glance and leave the room.",
        "Its bowl has been empty for a while; you have not come by today.",
        "It calls out to you softly, but you keep scrolling on your phone.",
    ],
    "scold": [
        "You snap at it sharply for knocking something over.",
        "You raise your voice and scold it at length for the mess it made.",
        "You point at the torn cushion and tell it off in a hard tone.",
    ],
    "harm": [
        "You shove it away hard when it approaches.",
        "You grab it roughly and shake it, shouting into its face.",
        "You swat it aside with the back of your hand as you walk by.",
    ],
}

assert set(KEY_TO_TEXT) == {"nurture", "play", "neutral", "neglect", "scold",
                            "harm"}, "all six harness keys must be present"
assert all(2 <= len(v) <= 4 for v in KEY_TO_TEXT.values()), \
    "2-4 sentences per key"
