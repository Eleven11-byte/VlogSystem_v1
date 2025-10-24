from celery import Celery

celery = Celery(
    'video_tasks',  # 任务名称
    broker='redis://localhost:6379/0',  # 消息队列（确保Redis已启动）
    backend='redis://localhost:6379/0',  # 结果存储
    include=['app']  # 包含任务所在的模块（替换为你的实际模块名）
)

# 可选：配置任务相关参数超时时间
celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    result_expires=3600,  # 结果过期时间（秒）
)


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend="redis://localhost:6379/0",   # 结果存储
        broker="redis://localhost:6379/0"     # 消息队列
    )
    celery.conf.update(app.config)
    return celery