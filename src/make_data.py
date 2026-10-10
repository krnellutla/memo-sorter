"""Build train.csv from templates and test.csv from hand-written notes.

The test notes are written separately from the templates, with different wording,
so the eval measures generalisation and not template memorisation.
"""
import csv
import itertools
import random
from pathlib import Path

random.seed(7)
OUT = Path(__file__).resolve().parent.parent / "data"

TOPICS = [
    "the pricing page", "the garden fence", "my resume", "the Q3 report", "the kitchen light",
    "the podcast intro", "the retrieval bot", "dad's birthday", "the tax forms", "the bike tyre",
    "the weekly newsletter", "the landing page", "the cat vet", "the fridge filter", "the demo video",
    "the dataset labels", "the passport renewal", "the car insurance", "the team offsite", "the logo",
]
PEOPLE = ["Priya", "Sam", "Mom", "Dr. Chen", "Jordan", "the landlord", "Alex", "Rita", "Marcus", "the plumber"]
TIMES = ["tomorrow", "on Friday", "next Tuesday", "at 5pm", "this weekend", "before lunch", "by the 15th", "tonight"]
FEELINGS = ["tired", "pretty good", "anxious", "proud", "restless", "calm", "frustrated", "grateful"]

TEMPLATES = {
    "idea": [
        "What if we made {t} interactive",
        "Idea: turn {t} into a small side project",
        "Could try using a tiny model for {t}",
        "Maybe {t} should be a weekly series",
        "Wouldn't it be cool if {t} explained itself",
        "Concept for {t}: show a before and after",
        "New angle on {t}, let users vote on it",
        "What about a game built around {t}",
        "Thought: {t} could run entirely offline",
        "Experiment: rewrite {t} from scratch and compare",
    ],
    "todo": [
        "Finish {t}",
        "Need to update {t} and send it over",
        "Todo: clean up {t}",
        "Fix {t} before the weekend",
        "Write the first draft of {t}",
        "Buy supplies for {t}",
        "Review {t} and leave comments",
        "Still have to sort out {t}",
        "Add {t} to the checklist and tick it off",
        "Start on {t} and ship something small",
    ],
    "reminder": [
        "Call {p} {tm}",
        "Remind me to email {p} {tm}",
        "Pick up the parcel {tm}",
        "Appointment with {p} {tm}",
        "Don't forget to pay the bill {tm}",
        "Meeting with {p} {tm}",
        "Renew the subscription {tm}",
        "Reply to {p} {tm}",
        "Bring the documents to {p} {tm}",
        "Water the plants {tm}",
    ],
    "journal": [
        "Felt {f} today after working on {t}",
        "Long day. Spent hours on {t} and I am {f}",
        "Talked to {p} about {t}, feeling {f}",
        "Went for a walk and thought about {t}. Mostly {f}",
        "Today was {f}. {t} did not go as planned",
        "I keep coming back to {t}, honestly feeling {f}",
        "Slept badly, woke up {f}, then {t} took all morning",
        "Had dinner with {p}. We laughed a lot, I feel {f}",
        "Looking back at this week, {t} made me {f}",
        "Quiet evening, a bit {f}, wrote down thoughts on {t}",
    ],
}


def fill(template: str) -> str:
    return template.format(
        t=random.choice(TOPICS), p=random.choice(PEOPLE),
        tm=random.choice(TIMES), f=random.choice(FEELINGS),
    )


def build_train(per_class: int = 90):
    rows = []
    for label, temps in TEMPLATES.items():
        seen = set()
        while len(seen) < per_class:
            seen.add(fill(random.choice(temps)))
        rows += [(text, label) for text in seen]
    random.shuffle(rows)
    return rows


# Hand-written, different phrasing from the templates.
TEST = {
    "idea": [
        "A browser extension that rewrites error messages in plain English",
        "Train a model on my own commit messages and see what it learns",
        "Podcast where each episode dissects one failed experiment",
        "Use image embeddings to find duplicate photos on my laptop",
        "Build a tiny voice assistant that never leaves the house network",
        "Compare three small models on the same fifty questions and publish the table",
        "A recipe app that scales for however many mouths are at the table",
        "Pair a camera with a tiny model to count birds at the feeder",
        "Newsletter written entirely from my own bookmarks",
        "Chess opponent that explains its worst moves out loud",
        "Maybe I could teach a small model to spot sarcasm in my emails",
        "A wall display that shows what my home server is thinking",
    ],
    "todo": [
        "Rewrite the README so a stranger can run it",
        "Submit the expense claim",
        "Buy a new SSD for the home server",
        "Replace the batteries in the smoke alarms",
        "Book flights for December",
        "Reorganise the dataset folder and delete the duplicates",
        "Ship version two of the notes page",
        "Order printer ink and a spare cable",
        "Take the old monitor to recycling",
        "Back up the photos to the external drive",
        "Fill in the school form and scan it",
        "Update dependencies and rerun the tests",
    ],
    "reminder": [
        "Dentist, Thursday 9:30",
        "Anniversary dinner on the 22nd, book a table",
        "Trash goes out Monday night",
        "Return the library books by Saturday",
        "Call mom after her appointment",
        "Standup moved to 10:15 tomorrow",
        "Car service is due next Wednesday",
        "Send Ravi the invoice before month end",
        "Parent teacher meeting at 4 on Friday",
        "Domain renews in two weeks",
        "Pick up dry cleaning on the way home",
        "Flight check-in opens tomorrow at noon",
    ],
    "journal": [
        "Rainy morning. Made coffee, sat with the window open and felt oddly peaceful",
        "I doubt myself more than I let on, especially after demos",
        "Good call with an old friend, I miss those conversations",
        "Spent the whole afternoon debugging and ended up laughing at my own mistake",
        "Not my best day. Snapped at someone and regret it",
        "Finished a long run and felt invincible for about ten minutes",
        "Proud that I stuck with the project even when nobody was watching",
        "Strange mood tonight, nostalgic for no reason",
        "The house was quiet and for once I did not reach for my phone",
        "Feeling behind on everything, but today I made small progress",
        "Watched the sunset from the balcony, grateful for ordinary days",
        "Cried a little at a film and then felt lighter",
    ],
}


def write(path: Path, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    train = build_train()
    test = [(t, lab) for lab, texts in TEST.items() for t in texts]
    random.shuffle(test)
    train_texts = {t for t, _ in train}
    leaked = [t for t, _ in test if t in train_texts]
    assert not leaked, leaked
    write(OUT / "train.csv", train)
    write(OUT / "test.csv", test)
    print(f"train={len(train)} test={len(test)}")
