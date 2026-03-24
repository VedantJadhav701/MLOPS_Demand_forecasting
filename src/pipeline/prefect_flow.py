from prefect import flow, task
from src.pipeline.train_pipeline import run_pipeline


@task
def train_task():
    run_pipeline()


@flow(name="demand_forecasting_pipeline")
def mlops_flow():
    train_task()


if __name__ == "__main__":
    mlops_flow()