# Jazz 2024-25: shot quality and shot-making, one page

*For: coaching, scouting, performance. Data: public shot-chart data, 2024-25 regular season. Model: expected points per shot from location and shot type; no defender tracking. Method and caveats in the repo README.*

## Bottom line
1. **We took low-value shots**, 27th of 30 in expected points per shot (1.066 vs a league range of 1.04-1.12). The driver is a low rim rate.
2. **Our shooting was exactly as good as the shots deserved** (0.0 points per 100 shots over expected, 18th), but unevenly: good inside and in the corners, poor on above-the-break threes and mid-range.
3. **The shots we allowed were fine; how they were made was not.** Opponent shot quality ranked 9th best, yet opponents shot **+4.6 points per 100 above expectation, 29th of 30** (about 350 points, or 4 per game). It was concentrated at the rim, non-restricted paint and mid-range.

## Shot mix: where the value gap comes from
| Zone | Jazz share of attempts | League share | Opponents' shots vs Jazz |
|---|---|---|---|
| Rim | **23.2%** | 27.9% | 24.2% |
| Paint (non-restricted) | 24.8% | 20.3% | 23.1% |
| Mid-range | 7.2% | 9.8% | 8.1% |
| Corner 3 | 12.8% | 10.8% | 10.3% |
| Above-break 3 | 32.0% | 31.2% | 34.3% |

About 4.5 points of attempt share sit in non-restricted-paint twos instead of at the rim, and those are worth far less (0.89 vs 1.33 points per shot league-wide).

## Shot-making by zone (points over expected per 100 shots)
| Zone | Jazz | Jazz opponents |
|---|---|---|
| Rim | +5.4 | **+8.0** |
| Paint (non-restricted) | +5.1 | **+10.5** |
| Mid-range | **-8.2** | **+8.3** |
| Corner 3 | +3.1 | +1.9 |
| Above-break 3 | **-7.3** (about 170 points on 2,321 attempts) | -2.1 |

## What to trust
- **Team-level year-to-year persistence** (30 teams, three season pairs): shot quality allowed r = 0.80, shot quality taken 0.59, shot-making 0.51, **shot-making allowed 0.41**. The allowed figure is partly real (contest quality and rim deterrence are not in this data) and partly luck. I would treat it as a lead for film and tracking review, not a verdict.
- **Individual shot-making is noisy.** Estimates below are shrunk toward the league mean; the bar is a 90% interval. Only four players' intervals exclude zero.

| Player (shots) | Points over expected / 100 (90% interval) |
|---|---|
| Brice Sensabaugh (599) | **+9.1** (+2.1, +16.1) |
| Collin Sexton (867) | **+6.7** (+1.1, +12.3) |
| John Collins (531) | +4.3 (-2.2, +10.9) |
| Kyle Filipowski (518) | +4.1 (-2.9, +11.0) |
| Johnny Juzang (475) | +1.8 (-5.8, +9.4) |
| Walker Kessler (419) | +1.5 (-4.8, +7.9) |
| Jordan Clarkson (493) | +0.5 (-6.7, +7.6) |
| Svi Mykhailiuk (302) | -2.4 (-11.0, +6.2) |
| Keyonte George (919) | -3.4 (-9.3, +2.5) |
| Lauri Markkanen (700) | -4.5 (-11.0, +2.0) |
| Isaiah Collier (555) | **-7.4** (-13.9, -0.8) |
| Cody Williams (257) | **-13.0** (-21.7, -4.2), smallest sample; raw figure was -26 before shrinkage |

Injuries, role and minutes are not in the data; a low rating on few shots mostly means "unlucky or unmeasured", not "poor shooter".

## What I would do next
1. Pull contest/closest-defender data for opponent rim and paint attempts to test whether the +8 to +10 is defence or luck.
2. Size the prize for closing the 4.7-point rim-share gap: about 340 shots at roughly 0.40 expected points each (Jazz rim 1.28 vs non-restricted paint 0.88) is about 135 points a season. That is an upper bound, since not every paint shot can become a rim attempt.
3. Re-run with 2025-26 data once available; the pipeline scores new files with `scripts/score_shots.py`.
