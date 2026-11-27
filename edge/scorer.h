#ifndef MICRODUCK_SCORER_H
#define MICRODUCK_SCORER_H
#include <stdint.h>

enum { NSO_HISTORY = 6, NSO_CHOICES = 3 };
enum nso_behavior { NSO_DEFER = -1, NSO_TRACK, NSO_SEARCH, NSO_WAIT };
/* Six 5 Hz ticks, oldest first: missing=0, absent=1, visible=2.
 * fresh_owned must reflect snapshot freshness and autonomous ownership.
 * Mask bits correspond to TRACK, SEARCH, WAIT. Caller rechecks at dispatch.
 * Optional output receives three raw logits only on success. */
const char *nso_model_id(void);
int nso_select(const uint8_t history[NSO_HISTORY], unsigned mask,
               int fresh_owned, float logits[NSO_CHOICES]);
#endif
