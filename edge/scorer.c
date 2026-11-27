#include "scorer.h"

static const float scores[729][NSO_CHOICES] = {
#include "scores.inc"
};

const char *nso_model_id(void) {
    return "choice-v1/726023a92444dc6267ebe816188e998e70cac459";
}

int nso_select(const uint8_t history[NSO_HISTORY], unsigned mask,
               int fresh_owned, float logits[NSO_CHOICES]) {
    if (!history || !fresh_owned || !mask || (mask & ~7u))
        return NSO_DEFER;
    unsigned row = 0;
    for (int tick = 0; tick < NSO_HISTORY; ++tick) {
        if (history[tick] > 2)
            return NSO_DEFER;
        row = 3 * row + history[tick];
    }
    if (history[NSO_HISTORY - 1] == 0)
        return NSO_DEFER;
    int best = NSO_DEFER;
    for (int i = 0; i < NSO_CHOICES; ++i) {
        if (logits)
            logits[i] = scores[row][i];
        if ((mask & (1u << i)) && (best == NSO_DEFER || scores[row][i] > scores[row][best]))
            best = i;
    }
    return best;
}
