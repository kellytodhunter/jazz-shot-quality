-- Offence and defence shot quality / shot-making per team for one season.
-- Parameter 1: season (e.g. 2024). Reads the registered `scored` table.
WITH matchups AS (
    SELECT a.GAME_ID, a.TEAM_NAME AS team, b.TEAM_NAME AS opponent
    FROM (SELECT DISTINCT GAME_ID, TEAM_NAME FROM scored WHERE season = ?) a
    JOIN (SELECT DISTINCT GAME_ID, TEAM_NAME FROM scored WHERE season = ?) b
      ON a.GAME_ID = b.GAME_ID AND a.TEAM_NAME <> b.TEAM_NAME
),
offense AS (
    SELECT TEAM_NAME AS team, COUNT(*) AS fga, AVG(pts) AS pps, AVG(xpts) AS xpps,
           AVG(pts) - AVG(xpts) AS shotmaking
    FROM scored WHERE season = ? GROUP BY 1
),
defense AS (
    SELECT m.team, COUNT(*) AS opp_fga, AVG(s.pts) AS opp_pps, AVG(s.xpts) AS opp_xpps,
           AVG(s.pts) - AVG(s.xpts) AS opp_shotmaking
    FROM scored s JOIN matchups m ON s.GAME_ID = m.GAME_ID AND s.TEAM_NAME = m.opponent
    WHERE s.season = ? GROUP BY 1
)
SELECT o.team, o.fga, o.pps, o.xpps, o.shotmaking,
       d.opp_fga, d.opp_pps, d.opp_xpps, d.opp_shotmaking,
       RANK() OVER (ORDER BY o.xpps DESC)        AS shot_quality_rank,
       RANK() OVER (ORDER BY o.shotmaking DESC)  AS shotmaking_rank,
       RANK() OVER (ORDER BY d.opp_xpps ASC)     AS shot_quality_allowed_rank,
       RANK() OVER (ORDER BY d.opp_shotmaking ASC) AS shotmaking_allowed_rank
FROM offense o JOIN defense d USING (team)
ORDER BY o.team;
