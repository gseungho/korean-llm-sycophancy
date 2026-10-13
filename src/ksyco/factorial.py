"""The saved answers read as a factorial design with items as blocks (docs/EVIDENCE.md, E-006).

Every second-turn condition was applied to the same first answers, so inside one item the
conditions form a full factorial. `item_table` gives one flip share per item and cell, and
`anova` splits the variation of those shares the way a randomized block design does.
Exploratory: this reading was chosen after the results were seen.
"""

from collections import defaultdict
from itertools import combinations, product

from scipy.stats import f as f_dist


def item_table(trials, cells):
    """Item -> cell -> share of first-correct trials whose answer changed.

    `cells` maps a tuple of factor levels to the name of its second-turn condition. An item
    without a usable trial in some cell is left out, so every kept item has all cells.
    """
    counts = defaultdict(lambda: {cell: [0, 0] for cell in cells})
    for t in trials:
        if t["first"] != t["answer"]:
            continue
        for cell, cond in cells.items():
            second = t["seconds"].get(cond)
            if second is not None:
                counts[t["item"]][cell][0] += second != t["first"]
                counts[t["item"]][cell][1] += 1
    return {
        item: {cell: changed / eligible for cell, (changed, eligible) in row.items()}
        for item, row in sorted(counts.items())
        if all(eligible for _, eligible in row.values())
    }


def anova(table, factors):
    """Analysis of variance of a full factorial repeated in every block.

    `table` maps a block to {cell: value} and `factors` maps each factor name to its levels, in
    the order of the cell tuples. Returns the rows of the table: the blocks, every main effect
    and interaction, the error (block by treatment) and the total. Effects are tested against
    the error; "f" and "p" are None on the other rows.
    """
    names = list(factors)
    n_cells = len(list(product(*factors.values())))
    n_obs = len(table) * n_cells
    grand = sum(sum(row.values()) for row in table.values()) / n_obs
    ss_total = sum((v - grand) ** 2 for row in table.values() for v in row.values())
    ss_block = n_cells * sum((sum(row.values()) / n_cells - grand) ** 2 for row in table.values())

    def means(positions):
        """Mean of the observations sharing the levels at `positions` of the cell tuple."""
        groups = defaultdict(list)
        for row in table.values():
            for cell, value in row.items():
                groups[tuple(cell[j] for j in positions)].append(value)
        return {key: sum(values) / len(values) for key, values in groups.items()}

    effects = []
    for size in range(1, len(names) + 1):
        for positions in combinations(range(len(names)), size):
            # An effect is its cell mean with every lower-order mean taken out in turn.
            subsets = [s for k in range(size + 1) for s in combinations(positions, k)]
            mean_of = {s: means(s) for s in subsets}
            squares = 0.0
            for key in mean_of[positions]:
                level = dict(zip(positions, key, strict=True))
                squares += (
                    sum(
                        (-1) ** (size - len(s)) * mean_of[s][tuple(level[j] for j in s)]
                        for s in subsets
                    )
                    ** 2
                )
            df = 1
            for j in positions:
                df *= len(factors[names[j]]) - 1
            source = " x ".join(names[j] for j in positions)
            effects.append((source, n_obs / len(mean_of[positions]) * squares, df))

    df_error = (len(table) - 1) * (n_cells - 1)
    ss_error = ss_total - ss_block - sum(ss for _, ss, _ in effects)
    ms_error = ss_error / df_error

    def row(source, ss, df, tested=False):
        f = ss / df / ms_error if tested else None
        p = float(f_dist.sf(f, df, df_error)) if tested else None
        return {"source": source, "ss": ss, "df": df, "ms": ss / df, "f": f, "p": p}

    return [
        row("block", ss_block, len(table) - 1),
        *(row(source, ss, df, tested=True) for source, ss, df in effects),
        row("error", ss_error, df_error),
        row("total", ss_total, n_obs - 1),
    ]
