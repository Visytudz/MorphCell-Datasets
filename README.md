# MorphCell data workspaces

This repository contains the preprocessing code and workspace documentation for the datasets used by MorphCell.

## Workspaces

- `colon/`: human colon cell preprocessing
- `wheat/`: wheat root cell preprocessing
- `redblood/`: red blood cell preparation
- `IntrA/`: IntrA point-cloud preparation
- `ShapeNet55-34/`: ShapeNet55/34 data organization

See the README in each directory for its data source, layout, scale conventions, and preprocessing workflow.

## Environment

Install the shared environment from the repository root:

```bash
uv sync
```