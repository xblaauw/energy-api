## Infrastructure
- [ ] Dockerfile
- [ ] Deployment config
- [ ] Deployment
- [ ] DNS
- [ ] Boilerplate such as tests etc

## Refactor: Extract Battery Optimization into Separate Package

### 1. Setup UV Workspace
- [ ] Convert root `pyproject.toml` to workspace configuration
  - Add `[tool.uv.workspace]` section
  - Define `members = ["packages/*"]`
- [ ] Create `packages/battery-optimizer/` directory structure
- [ ] Run `uv init --lib packages/battery-optimizer` to initialize new package
  - Set name: `battery-optimizer`
  - Set Python version: `>=3.12`
  - Add dependencies: `pandas`, `numpy`, `pulp`, `pydantic`

### 2. Create Package Structure
Create in `packages/battery-optimizer/src/battery_optimizer/`:
- [ ] `__init__.py` - Export main public API (`optimize_battery` function)
- [ ] `models.py` - Data models (BatterySpecs, GridLimits, EnergyDataPoint, etc.)
  - Move from `app/schemas/battery.py` or create simplified versions
  - Keep optimizer-focused models (no FastAPI Response models here)
- [ ] `validation.py` - Input validation logic
  - Move `_validate_and_parse_energy_data()` from `app/routers/battery.py:51-84`
  - Make it accept list of data points + validation params
- [ ] `core.py` - Core optimization solver
  - Move `_solve_battery_optimization()` from `app/routers/battery.py:87-212`
  - Return plain Python dict/dataclass instead of FastAPI response model
  - Keep PuLP logic isolated here

### 3. Update API Layer
- [ ] Update `app/routers/battery.py`:
  - Add import: `from battery_optimizer import optimize_battery`
  - Simplify endpoint to just call optimizer and format response
  - Keep only HTTP/FastAPI-specific logic (error handling, status codes)
- [ ] Update `app/schemas/battery.py`:
  - Keep FastAPI-specific request/response models
  - Import battery specs from `battery_optimizer.models` if reusing

### 4. Configure Dependencies
- [ ] Add local package to API dependencies:
  - Run: `uv add --editable ./packages/battery-optimizer` from project root
  - This adds it to root `pyproject.toml` dependencies
- [ ] Verify `uv.lock` includes the local package correctly

### 5. Testing & Validation
- [ ] Test the optimizer package independently (can create simple test script)
- [ ] Test API endpoint still works with refactored code
- [ ] Verify all edge cases still handled (validation errors, optimization failures)
- [ ] Check that all imports resolve correctly

### 6. Documentation (Optional)
- [ ] Add README.md to `packages/battery-optimizer/`
- [ ] Document public API (function signatures, parameters, return types)
- [ ] Add usage examples for standalone package use