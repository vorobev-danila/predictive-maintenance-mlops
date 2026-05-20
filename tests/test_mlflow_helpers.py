from botocore.exceptions import ClientError

import pipeline


class FakeS3Client:
    def __init__(self, error_code=None):
        self.error_code = error_code
        self.created_buckets = []

    def head_bucket(self, Bucket):
        if self.error_code is None:
            return None
        raise ClientError({"Error": {"Code": self.error_code}}, "HeadBucket")

    def create_bucket(self, Bucket):
        self.created_buckets.append(Bucket)


def test_ensure_minio_bucket_creates_missing_bucket(monkeypatch):
    fake_client = FakeS3Client(error_code="404")
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minio")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "minio123")
    monkeypatch.setattr(pipeline.boto3, "client", lambda *args, **kwargs: fake_client)

    pipeline.ensure_minio_bucket("mlflow")

    assert fake_client.created_buckets == ["mlflow"]


def test_ensure_minio_bucket_does_not_create_existing_bucket(monkeypatch):
    fake_client = FakeS3Client()
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minio")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "minio123")
    monkeypatch.setattr(pipeline.boto3, "client", lambda *args, **kwargs: fake_client)

    pipeline.ensure_minio_bucket("mlflow")

    assert fake_client.created_buckets == []


def test_ensure_minio_bucket_reraises_unexpected_error(monkeypatch):
    fake_client = FakeS3Client(error_code="403")
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minio")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "minio123")
    monkeypatch.setattr(pipeline.boto3, "client", lambda *args, **kwargs: fake_client)

    try:
        pipeline.ensure_minio_bucket("mlflow")
    except ClientError as error:
        assert error.response["Error"]["Code"] == "403"
    else:
        raise AssertionError("Expected ClientError for unexpected MinIO error")
