from rich.columns import Columns
from rich.table import Table


def plot_table(title: str, items: list[dict], prompt_on_empty: str, width: int | None = None) -> Table:
    table = Table(title=title, show_header=True, header_style="bold blue", width=width)

    if len(items) > 0:
        for column_name in items[0]:
            table.add_column(column_name, overflow="fold")
        for item in items:
            table.add_row(*list(item.values()))
    else:
        print(prompt_on_empty)

    return table


def plot_nested_table(
    title: str,
    items,
    inner_table_key: str,
    prompt_on_empty: str,
    width: int | None = None,
    width_ratios=(8, 12, 20, 10, 50),
) -> Table:
    table = Table(title=title, show_header=True, header_style="bold blue", width=width)

    if len(items) > 0:
        # Add header to the outer table
        for idx, outer_column_name in enumerate(items[0].keys()):
            table.add_column(outer_column_name, overflow="fold", ratio=width_ratios[idx])

        for item in items:
            inner_table = Table(show_header=True, header_style="bold green")
            if len(item[inner_table_key]) > 0:
                # Add header to the inner table
                for idx, inner_column_name in enumerate(item[inner_table_key][0].keys()):
                    inner_table.add_column(inner_column_name, overflow="fold", ratio=width_ratios[idx])
                # Add rows to the inner table
                for exp in item[inner_table_key]:
                    inner_table.add_row(*list(exp.values()))

            # Add rows to the outer table
            table.add_row(*list(item.values())[:-1], Columns([inner_table]))
    else:
        print(prompt_on_empty)

    return table
