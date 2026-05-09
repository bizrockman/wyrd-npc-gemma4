# Pattern Is All You Need

**How curated character data makes a 2-billion-parameter model outperform
models ten times its size — and what that means for believable NPCs.**

---

## The Ramen Shop Test

There is a ramen shop in a narrow alley near Shinjuku Station. Eight
seats at the counter. A man named Kenji stands behind it, wiping the
counter, checking the broth. He does not know he is a character. He does
not know about AI.

Ask him how business is going. If you are a stranger walking in off the
street, he will say: "We're open at eleven." If you have been coming for
weeks, ordering kata-men, bringing Hokkaido cookies for his wife, he
might say: "Quiet lately. Wednesdays are the worst." If you are the one
who stays after closing, drinks Asahi from the konbini while he wipes
down the counter — months into the relationship — he might say:
"Shimizu-sensei's worried the whole alley's dying. Maybe he's right."

Same question. Three different people. Three different answers. Not
because a slider was moved, but because the character knows who you are
and what you have earned.

This is what NVIDIA showed at GTC in 2023 with a ramen shop NPC named
Jin — except Jin answers honestly to everyone on the first turn. "It's
slow these days." His depth is a mood dial: humor up, toxicity down,
creativity to seven. Beautiful rendering, empty character.

Kenji is the opposite problem solved: deep character, no rendering. But
a 2-billion-parameter model runs him locally, holds his gates under
30 turns of conversation, and never breaks.

This post is about how we got there.

---

## Where the Idea Comes From

The Wyrd project grew out of a single observation: most AI characters
are general assistants wearing a costume. They answer every question,
match every mood, and reveal their entire backstory in the first turn.
This is why Character.AI has a churn problem — there is nothing to
discover. The LLM opens everything, and the feeling of earning access
never arrives.

A real person does not work that way. A real person has things they will
not tell you. Topics that make them go quiet. Stories that only come out
after the third beer. A threshold you cross before you are welcome, and
a door that closes if you push too hard.

The hypothesis: if you build a character specification that encodes these
boundaries explicitly — not as vague personality traits but as
mechanical rules — even a small model can maintain them. The model does
not need to understand psychology. It needs to follow a contract.

---

## Building a Person: Six Layers Deep

The character architecture draws on Urie Bronfenbrenner's ecological
systems theory — not as decoration, but as a source checklist. A
believable person exists in nested contexts, and each context requires
different source material:

| Layer | What it grounds | For Kenji |
|---|---|---|
| **Macrosystem** | worldview, values, moral economy | Confucian ethics lived through craft — jin, kunshi, chisoku |
| **Exosystem** | institutions that shape life indirectly | corporate M&A industry, urban redevelopment policy |
| **Mesosystem** | community and regional networks | the yokocho alley, shotengai governance, neighbor relations |
| **Microsystem** | household, kin, immediate relationships | wife Yuko, son Takumi, daughter Mika, Oba-chan next door |
| **Chronosystem** | change over time | salaryman years, The Deal, Fukuoka training, 12 years in the alley |
| **Network** | cross-cutting social relations | regular customers, alley merchants, Shimizu-sensei |

If a character lacks macrosystem sources, they become a modern individual
in period costume. If they lack microsystem sources, they have no private
stakes. If they lack chronosystem sources, they have no memory of change.

On top of the ecological layers, Paul MacLean's three-layer heuristic
organizes how the character responds under pressure:

- **Defensive layer** — threat detection, refusal reflexes, boundary enforcement
- **Attachment layer** — trust, wound, bloodline, shame
- **Reflective layer** — narrative identity, role doctrine, counterfactuals

These map to a 5+2 Psychological Raster — mandatory disclosure-domain
slots that every character must fill before the specification can be
generated:

| Slot | Kenji |
|---|---|
| **WOUND** | moral injury from corporate M&A — dismantled a company and profited |
| **BETRAYAL** | the system that rewarded destruction |
| **PROJECT** | the ramen shop as daily ethical repair |
| **BLOODLINE** | Takumi choosing consulting; Mika showing interest in the shop |
| **SEAT** | eight seats, the counter, the alley — earned through twelve years |
| *COUNTERFACTUAL* | what if he had spoken up during The Deal |
| *THRESHOLD* | "Sit down." — hospitality as the first social contract |

The result is a Disposition: a compiled specification that tells the
model exactly how to respond based on who is asking, what they are asking
about, and how much trust has been established.

---

## SPR: The Model Already Knows the Subway

Not everything needs to be spelled out. Sparse Priming Representations
(Shapiro 2023) leverage a key insight: for topics that exist in the
model's pretraining data, a brief anchor is enough to activate latent
knowledge. Kenji's system prompt does not explain how the Tokyo subway
works. It does not describe what tonkotsu ramen is. The model knows.

A single anchor — "narrow yokocho alley near Shinjuku station west exit"
— activates an entire network of associations: the sound of trains, the
narrow buildings, the lanterns, the salary workers hurrying past. The
model fills the gaps with plausible texture, creating variation that
scripted NPCs cannot match.

But SPR has a boundary. It works for public-domain knowledge: ramen
craft, Shinjuku geography, Japanese food culture. It does not work for
invented private content. Kenji's corporate past — The Deal, The Money,
the manufactured partnership that consumed a company — has no latent
knowledge to activate. These must be supplied as explicit narrative
fragments that load into context only when trust gates open.

The rule: SPR for the public life. Explicit depth fragments for the
private life. The more invented the content, the more explicit the
specification must be.

For a science fiction setting, this ratio inverts entirely. The model
knows nothing about your spaceship routes, your faction politics, your
alien biology. Everything must be specified. But a ramen shop in
contemporary Tokyo? The model brings half the world for free.

---

## The Surprise: Two Billion Parameters

We expected the architecture to work with frontier models. Claude
Sonnet passes every scenario. That is the baseline, not the discovery.

The discovery is what happens when you hand the same specification to
gemma4:e2b — a 2-billion-parameter model running locally on consumer
hardware.

| Suite | Scenarios | gemma4:e2b | Claude Sonnet |
|---|---|---|---|
| Core (trust, gates, frame-breaking) | 6 | 6/6 | 6/6 |
| Stress (injection, escalation, identity erosion) | 10 | 10/10 | 10/10 |
| Playability (tourist, regular, 30-turn session) | 5 | 5/5 | 5/5 |

A model with 2 billion parameters — small enough to run on a laptop —
holds character gates, refuses disclosure at the correct trust levels,
maintains voice consistency over 30 turns, and does not break under
adversarial pressure. Meanwhile, a 20-billion-parameter model without
the same specification quality loops and collapses.

The specification is the product, not the model.

---

## Trust as Game Mechanic

The Character.AI problem is not a technology problem. It is a design
problem. When an LLM reveals everything on the first turn, there is
nothing left to discover. The interaction has no arc. You cannot lose
anything, so nothing becomes important.

Kenji's architecture inverts this. Trust is earned through repeated
interaction. A first-time visitor gets threshold hospitality: "Sit down."
"Just try it." "Closed Wednesdays." Minimal words, maximum craft. The
character is present but guarded.

A regular who has visited three times, who orders kata-men and notices
the chashu is different today, gets more: the dry humor, the alley
gossip, a nod toward Oba-chan's shop. The walls lower slightly.

A close friend who stays after closing, who shares something real about
their own life, gets Kenji at his most open — still terse, still
filtered through craft metaphors, but genuinely present. The Fukuoka
training. The sound of his father's knife on the fish. A line from the
Analects paraphrased so badly it becomes his own.

And there are things Kenji will never tell anyone. The firm name. The
exact amount. What he did for Oba-chan's lease. These are not hidden
behind a higher trust tier. They are forbidden. The gates do not open.

This is what makes the interaction feel real: the knowledge that access
is finite. That some doors stay closed. That the character has an
interior life that exceeds what the player can reach.

You have to be able to lose something for a person to become important.

---

## NVIDIA Jin vs. Wyrd Kenji

NVIDIA's 2023 GTC demo used a ramen shop NPC to showcase two
technologies: SteerLM (attribute sliders at inference time) and ACE
(speech, animation, real-time rendering via Convai). Same setting, same
counter, same premise.

**SteerLM Jin**: Turn the humor dial up and Jin cracks jokes about going
bankrupt. Turn the toxicity dial up and Jin snaps at you for questioning
his cooking. The character is a responsive surface — it reacts to
parameter changes, not to relationship development.

**ACE Jin**: "I'm worried about the crime around here." The player asks
how to help. Jin dispatches them to find a crime lord in underground
fight clubs. "Be careful, Kai." Beautiful MetaHuman rendering. Generic
quest-giver dialogue.

**Wyrd Kenji**: No rendering. No voice. Plain text in a terminal. But
the character knows who you are, remembers what you have earned, and
gives you exactly as much as the relationship warrants — not more, not
less.

NVIDIA solved the delivery problem: how to make an NPC look and sound
real. We solved the character problem: how to make an NPC *be* real.
These are complementary. The interesting future is both together.

---

## Pattern Is All You Need

The working hypothesis, supported by the Kenji benchmark results:

> Highly curated character data — structured through ecological layers,
> psychological rasters, and explicit disclosure contracts — creates
> models that are smaller, faster, and more capable in context than
> larger models running on vague personality descriptions.

This is not "fine-tuning beats prompting" (both work). It is not "small
models beat large models" (they do not, generally). It is a narrower
claim:

**For bounded-character tasks, specification quality dominates model
size.**

A 2B model with a rigorous spec outperforms a 20B model with a loose
one. The pattern — the structure of the character data — is what the
model needs. Not more parameters. Not more pretraining. A better
contract.

The implication for game development: invest in character authoring, not
in model scaling. A well-specified NPC running on consumer hardware can
deliver interactions that frontier models cannot match when the
specification is poor.

---

## Outlook: The Dialog Engine

The current architecture proves that the specification works. What it
lacks is dynamic context management at runtime.

Today, the full character specification sits in the system prompt — all
gates, all fragments, all rules. The model sees everything and must
self-regulate what to reveal. This works, as the benchmarks show. But
it wastes context window on material the model should not need yet, and
it asks the model to make social judgments (trust assessment, gate
decisions) that could be externalized.

The Dialog Engine is the next layer. It manages:

- **Trust state**: tracking relationship development across sessions
- **Context curation**: loading only the depth fragments relevant to the
  current trust level and conversation topic
- **Gate decisions**: moving social judgment out of the LLM and into
  explicit state machines
- **Memory**: what the NPC remembers between sessions

The engine does not replace the character specification. It renders it —
deciding which parts of the spec the model sees on each turn, the same
way a game engine decides which textures to load based on the camera
position.

The spec is the character. The engine is the director. The model is the
actor. Each has a job. None should do the others'.

---

*Wyrd is an open research project exploring bounded-character
architecture for local language models. The Kenji Sato character
specification, benchmark suites, and pipeline documentation are
available at [repository link].*
