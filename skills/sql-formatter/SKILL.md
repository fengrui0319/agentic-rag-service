---
name: sql-formatter
description: Format messy SQL queries for readability. Use when the user asks to format, clean up, or prettify SQL.
---

# SQL Formatter

When formatting SQL:

1. Put major clauses on separate lines:
   - SELECT
   - FROM
   - JOIN
   - WHERE
   - GROUP BY
   - ORDER BY

2. Indent selected columns consistently.

3. Put each JOIN on its own line.

4. Indent AND / OR conditions under WHERE.

5. Use uppercase SQL keywords.

6. Do not change the meaning of the query.

## Example

Input:

```sql
select id,name from users where active=1 and age>18 order by name;
```

Output:

```sql
SELECT
    id,
    name
FROM users
WHERE active = 1
    AND age > 18
ORDER BY name;
```