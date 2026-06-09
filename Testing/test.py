import torch
import math
import pandas as pd
import matplotlib.pyplot as plt


def calculate_anomaly_scores(
    model,
    events,
    event_to_id,
    context_length=10
):
    model.eval()

    results = []

    with torch.no_grad():

        for idx in range(1, len(events)):

            # Build context
            start = max(0, idx - context_length)

            context_events = events[start:idx]

            actual_event = events[idx]

            context_ids = [
                event_to_id[e]
                for e in context_events
            ]

            actual_id = event_to_id[actual_event]

            x = torch.tensor(
                context_ids,
                dtype=torch.long
            ).unsqueeze(0)

            probs = torch.softmax(
                model(x),
                dim=-1
            )

            last_probs = probs[0, -1]

            actual_prob = (
                last_probs[actual_id]
                .item()
            )

            anomaly_score = -math.log(
                actual_prob + 1e-10
            )

            results.append({
                "position": idx,
                "context": " ".join(context_events),
                "actual_event": actual_event,
                "probability": actual_prob,
                "anomaly_score": anomaly_score
            })

    return pd.DataFrame(results)


def plot(df_scores) :
    plt.figure(figsize=(14,5))

    plt.plot(
        df_scores["position"],
        df_scores["anomaly_score"]
    )

    plt.xlabel("Log Position")
    plt.ylabel("Anomaly Score")
    plt.title(
        "Log Anomaly Detection Timeline"
    )

    plt.grid(True)

    plt.show()