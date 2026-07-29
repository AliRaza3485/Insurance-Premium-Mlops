# Architecture & Design Decisions

## Why XGBoost over Linear models?

Tested 4 models (Linear Regression, Ridge, Lasso, XGBoost). XGBoost won on all three metrics (MAE, RMSE, R²), primarily because it captures the non-linear interaction between smoking status and BMI discovered during EDA — a pattern linear models can't represent without manual feature engineering matching that exact interaction.

## Why an engineered smoker × BMI interaction feature?

EDA revealed that high BMI only significantly increases charges *when combined with smoking* — BMI alone had a weak correlation (0.20) with charges. Multiplying the binary `smoker` flag by `bmi` creates a feature that's zero for non-smokers and equal to BMI for smokers, letting models directly learn this compound effect.

## Why one-hot encoding for region but binary mapping for sex/smoker?

`sex` and `smoker` are binary categories with no ordinal relationship — a simple 0/1 mapping is sufficient. `region` has 4 categories with no natural order; one-hot encoding avoids implying a false ranking (e.g., northeast > southwest) that a numeric label encoding would introduce.

## Why MLflow with SQLite backend instead of file-based tracking?

Newer MLflow versions deprecated file-based (`./mlruns`) tracking in favor of a database backend. SQLite was chosen for simplicity — no external database server needed for a project at this scale.

## Why manual Docker Hub push instead of fully automating deployment in CI/CD?

The current CI/CD pipeline runs tests and verifies the Docker build succeeds on every push. Pushing to Docker Hub and pulling on the EC2 instance are currently manual steps, done deliberately to first validate the full deployment flow end-to-end before automating it. A natural next step would be adding Docker Hub push and remote deployment steps to the GitHub Actions workflow using repository secrets.

## Why EC2 over ECS/Fargate or Elastic Beanstalk?

EC2 was chosen for learning purposes — it offers the most direct, transparent control over the deployment environment and is the most commonly referenced AWS compute service in ML engineering job descriptions.

## Why keep statistical outliers in the training data?

Investigation showed the outliers (high-charge non-smokers) were valid customers, not data errors — their elevated charges correlated with age, not BMI as initially hypothesized. Removing them would train the model to ignore exactly the high-risk customers insurance pricing needs to predict accurately.