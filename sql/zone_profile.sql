-- Shot mix and efficiency by zone group, for a team's own shots and for the shots it allows.
-- Parameters: 1-2 season, 3 team name.
WITH opp AS (
    SELECT a.GAME_ID, b.TEAM_NAME AS shooter
    FROM (SELECT DISTINCT GAME_ID, TEAM_NAME FROM scored WHERE season = ?) a
    JOIN (SELECT DISTINCT GAME_ID, TEAM_NAME FROM scored WHERE season = ?) b
      ON a.GAME_ID = b.GAME_ID AND a.TEAM_NAME <> b.TEAM_NAME
    WHERE a.TEAM_NAME = ?
),
tagged AS (
    SELECT zone_group, pts, xpts, 'own'     AS side FROM scored WHERE season = ? AND TEAM_NAME = ?
    UNION ALL
    SELECT s.zone_group, s.pts, s.xpts, 'allowed' FROM scored s
        JOIN opp o ON s.GAME_ID = o.GAME_ID AND s.TEAM_NAME = o.shooter WHERE s.season = ?
    UNION ALL
    SELECT zone_group, pts, xpts, 'league'  FROM scored WHERE season = ?
)
SELECT side, zone_group, COUNT(*) AS shots,
       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY side) AS share,
       AVG(pts) AS pps, AVG(xpts) AS xpps
FROM tagged WHERE zone_group <> 'Backcourt'
GROUP BY side, zone_group ORDER BY side, zone_group;
