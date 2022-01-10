import logging
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
max_workers = 5
rabbitmq_publisher = {"exchange": "Event", "routing_key": "train_status.#", "exchange_type": 'topic', "durable": 'False'}
rabbitmq_consumer = {"exchange": "Event", "routing_keys": ["train.#"], "exchange_type": 'topic', "durable": 'False', "queue": "train_jobs_queue"}
platform_utility_configuration = {"debug": 1,
                                  "log_file": 'train.log',
                                  "mode": 1,
                                  "enable_db": True,
                                  "enable_redis": True,
                                  "enable_rabbitmq_consumer": True,
                                  "enable_rabbitmq_publisher": True,
                                  "redis_db": 0,
                                  "log_level": logging.DEBUG,
                                  "config_file_path": "config.json",
                                  "database": "ZestIot_AppliedAI"}
