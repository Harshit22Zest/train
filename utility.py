import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from datetime import datetime
import json
import logging
import time
import MySQLdb
import pandas as pd
import pika
import redis
from sqlalchemy import create_engine
from key_generators import make_key
from Logger import Logger
import sys


class RedisMgmt(object):
    def __init__(self, host='localhost', port=6379, db=0, password='', charset = 'utf-8', decode_responses = True, config_file_path='config.json'):
        print("*************************", config_file_path)
        config_json = json.loads(open(config_file_path).read())
        self.db_number = db
        self.r = redis.StrictRedis(host=host,
                                   port=port,
                                   db=db,
                                   charset=charset,
                                   decode_responses=decode_responses)


    def add_keys(self, vals, dev_id=''):
        """
        add a list of keys to user made list of keys
        :param vals: list of keys to be added (str)
        :param dev_id: device id (str) (default: str)
        :return: length of list after adding
        """
        if dev_id != '':
            return self.r.rpush(dev_id + '.keys', *vals)
        return self.r.rpush('keys', *vals)

    def rem_key(self, val, dev_id=''):
        """
        remove a key from the user made list of keys
        :param val: key to be deleted (str)
        :param dev_id: device id (str) (default: str)
        :return: number of removed keys
        """
        if dev_id != '':
            return self.r.lrem(dev_id + '.keys', 0, val)
        return self.r.lrem('keys', 0, val)

    def get_all_keys(self, dev_id=''):
        """
        gets all keys from user made list of keys
        :param dev_id: device id (str) (default: str)
        :return: all keys (list)
        """
        if dev_id != '':
            return self.r.lrange(dev_id + '.keys', 0, -1)
        return self.r.lrange('keys', 0, -1)

    def get_val(self, key, dev_id=''):
        """
        get key value for dev_id from redis
        :param key: like lat, lon, etc. (str)
        :param dev_id: device id (str) (default: str)
        :return: value of key (str)
        """
        if dev_id != '':
            return self.r.get(dev_id + '.' + key)
        return self.r.get(key)

    def set_val(self, key, val, dev_id=''):
        """
        set val of key of dev_id in redis
        :param key: key (str)
        :param val: value (int/float/str)
        :param dev_id: device id (str) (default: str)
        :return: True/False - success/failure (bool)
        """
        # keys = self.get_all_keys(dev_id=dev_id)
        # if key not in keys:
        #     self.add_keys([key], dev_id=dev_id)
        if dev_id != '':
            return self.r.set(dev_id + '.' + key, val)
        return self.r.set(key, val)

    def check_val(self, key, dev_id=''):
        """
        checks if key has a value stored in redis
        :param key: key (str)
        :param dev_id: device id (str) (default: '')
        :return: True if exists, otherwise False
        """
        val = self.get_val(key, dev_id)
        if val:
            return True
        return False

    def del_key(self, keys, dev_id=''):
        """
        deletes key from redis
        :param
        :return: number of keys deleted
        """
        if dev_id != '':
            keys = [dev_id + '.' + key for key in keys]
        return self.r.delete(*keys)

    def set_json(self, key, json):
        """
        adds json to redis
        :param key: key
        :param json: dict
        :return: True/False - success/failure (bool)
        """
        self.r.hmset(key, json)

    def get_json(self, key):
        """
        gets json from redis
        :param key: key (str)
        :return: dict
        """
        self.r.hgetall(key)

    @staticmethod
    def cache_it(key_prefix=None, timeout=None):
        """
        :param timeout: in seconds
        :param key_prefix:
        :param redis_connection: redis connection
        :return: cached heavy computation result
        """

        def cache_it_decorator(func):
            cache_key_prefix = key_prefix or func.__name__

            def cache_it_wrapper(*args, **kwargs):
                ut = kwargs.pop('utility')
                try:
                    redis_connection = ut.master_redis
                except AttributeError:
                    redis_connection = None

                if redis_connection is None:
                    redis_connection = RedisMgmt(db=12)
                delete_this = kwargs.pop('deleteThis', False)
                dummy_args = list(args)
                dummy_args.extend(kwargs.values())
                cache_key = "{}:{}".format(cache_key_prefix, make_key(*dummy_args))
                try:

                    if delete_this:
                        redis_connection.r.delete(cache_key)
                        return None

                    result = redis_connection.r.get(cache_key)

                    if not result:
                        result = func(*args, **kwargs)

                        redis_connection.r.set(cache_key, result, ex=timeout)
                except ConnectionError as err:
                    # logger.fatal(err)
                    if delete_this:
                        return None
                    return func(*args, **kwargs)
                return result

            return cache_it_wrapper

        return cache_it_decorator



class Utility(object):

    def __init__(self, configuration=None, rabbitmq_consumer=None, rabbitmq_publisher=None):
        if not configuration:
            self.configuration = {"debug": 0,
                                  "log_file": None,
                                  "mode": 1,
                                  "enable_db": False,
                                  "enable_redis": False,
                                  "enable_rabbitmq_consumer": False,
                                  "enable_rabbitmq_publisher": False,
                                  "redis_db": 0,
                                  "log_level": logging.DEBUG,
                                  "config_file_path": "config.json",
                                  "database": "ZestIot_AppliedAI"}
        else:
            self.configuration = configuration
        self.logger = Logger(self.configuration.get("debug"), self.configuration.get("mode"), self.configuration.get("log_file"), self.configuration.get("log_level"), config_file_path=self.configuration.get("config_file_path"))
        self.enable_db = self.configuration.get("enable_db")
        self.enable_redis = self.configuration.get("enable_redis")
        self.redis_db = self.configuration.get("redis_db")
        self.enable_rabbitmq_consumer = self.configuration.get("enable_rabbitmq_consumer")
        self.enable_rabbitmq_publisher = self.configuration.get("enable_rabbitmq_publisher")
        self.config_file_path = self.configuration.get("config_file_path")
        self.debug = self.configuration.get("debug")
        self.mode = self.configuration.get("mode")
        self.database = self.configuration.get("database")
        self.sql_engine = None
        self.tag = ""
        try:
            self.config_json = json.loads(open(self.config_file_path).read())
            self.config_json = self.config_json.get(self.config_json.get("env"))
        except Exception as e:
            self.loginfo("error reading config file: " + str(e), 'error')
        if self.enable_db:
            self.__init_db()
        if self.enable_redis:
            self.master_redis = RedisMgmt(db=self.redis_db, config_file_path=self.config_file_path)
        if self.enable_rabbitmq_consumer and rabbitmq_consumer:
            self.rabbitmq_consumer = rabbitmq_consumer
            self.get_rabbitmq_consumer(rabbitmq_consumer.get("exchange"), rabbitmq_consumer.get("queue"), rabbitmq_consumer.get("routing_keys"))
        if self.enable_rabbitmq_publisher and rabbitmq_publisher:
            self.rabbitmq_publisher = rabbitmq_publisher
            self.get_rabbitmq_publisher()

    def __init_db(self):
        try:
            self.db_read = MySQLdb.connect(host=self.config_json[self.database + "_mysql_read"]["host"],
                                           user=self.config_json[self.database + "_mysql_read"]["user"],
                                           passwd=self.config_json[self.database + "_mysql_read"]["passwd"],
                                           db=self.config_json[self.database + "_mysql_read"]["db"])
            self.db_read.autocommit(True)
            self.db_write = MySQLdb.connect(host=self.config_json[self.database + "_mysql_write"]["host"],
                                            user=self.config_json[self.database + "_mysql_write"]["user"],
                                            passwd=self.config_json[self.database + "_mysql_write"]["passwd"],
                                            db=self.config_json[self.database + "_mysql_write"]["db"])
            self.db_write.autocommit(True)
            self.sql_engine = create_engine("mysql+mysqldb://" + str(self.config_json[self.database + "_mysql_write"]["user"]) + ":" + self.config_json[self.database + "_mysql_write"]["passwd"] + "@" + str(self.config_json[self.database + "_mysql_write"]["host"]) + "/" + self.config_json[self.database + "_mysql_write"]["db"], pool_recycle=3600, pool_pre_ping=True, pool_size=15, max_overflow=10)

        except Exception as e:
            self.loginfo("error in connecting to mysql server: " + str(e), 'error')

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if key in ('debug', 'mode'):
            self.logger.__setattr__(key, value)

    def query_database(self, sql, **kwargs):
        try:
            self.close_connection()
        except Exception as e_:
            self.loginfo("Connection reestablished")
        self.__init_db()
        conn = self.db_read
        cursor = conn.cursor()
        self.loginfo("DB query: " + str(datetime.now()) + " Read Query: " + sql, 'warn')
        try:
            start_time = time.time()
            cursor.execute(sql)
            rows = cursor.fetchall()
            no_of_rows = cursor.rowcount
            time_diff = time.time() - start_time
            self.loginfo("Success, NoOfRows: " + str(no_of_rows) + " timeTaken= " + str(time_diff) + "secs", 'warn')
            return rows, no_of_rows
        except Exception as e:
            self.loginfo("Error in executing sql: {}".format(sql), 'error')
            self.loginfo('Error details: {}'.format(e), 'error')
            if "Lost connection to MySQL server during query" in str(e) or "MySQL server has gone away" in str(e) or "(0, '')" in str(e) or "(2013, '')" in str(e) or "(2006, '')" in str(e):
                try:
                    try:
                        self.close_connection()
                    except Exception as e_:
                        pass
                    self.__init_db()
                    conn = self.db_read
                    cursor = conn.cursor()
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                    no_of_rows = cursor.rowcount
                    self.loginfo("Success, NoOfRows: " + str(no_of_rows), 'debug')
                    return rows, no_of_rows
                except Exception as e:
                    self.loginfo("Error in retrying connecting to mysql server " + str(e), 'error')
            return [], 0

    def get_mapper(self, sql,var):
        """this is basically for select statement only where user has one field and using that has filter user want a another field (var) from same table """
        conn = self.db_read
        try:
            start_time = time.time()
            row = pd.read_sql(sql, con=conn)
            time_diff = time.time() - start_time
            self.loginfo("Time taken: {}, No of rows: {}".format(time_diff, len(row.index)), 'debug')
            return str(row[var][0])
        except Exception as e:
            self.loginfo("Error in executing sql: {}".format(sql), 'error')
            self.loginfo('Error details: {}'.format(e), 'error')
            if "Lost connection to MySQL server during query" in str(e) or \
                    "MySQL server has gone away" in str(e) or "(0, '')" in str(e):
                try:
                    try:
                        self.close_connection()
                    except Exception as e_:
                        pass
                    self.__init_db()
                    row = pd.read_sql(sql, con=self.db_read)
                    return str(row[var][0])
                except Exception as e:
                    self.loginfo("Error in retrying connecting to mysql server " + str(e), 'error')
            return None

    def query_database_df(self, sql, **kwargs):
        try:
            self.close_connection()
        except Exception as e_:
            self.loginfo("Connection reestablished")
        self.__init_db()
        conn = self.db_read
        self.loginfo("DB query: " + str(datetime.now()) + " Read Query: " + sql, 'warn')
        try:
            start_time = time.time()
            df = pd.read_sql(sql, con=conn)
            time_diff = time.time() - start_time
            self.loginfo("Time taken: {}, No of rows: {}".format(time_diff, len(df.index)), 'debug')
            return df
        except Exception as e:
            self.loginfo("Error in executing sql: {}".format(sql), 'error')
            self.loginfo('Error details: {}'.format(e), 'error')
            if "Lost connection to MySQL server during query" in str(e) or \
                    "MySQL server has gone away" in str(e) or "(0, '')" in str(e) or "(2013, '')" in str(e) or "(2006, '')" in str(e):
                try:
                    try:
                        self.close_connection()
                    except Exception as e_:
                        pass
                    self.__init_db()
                    df = pd.read_sql(sql, con=self.db_read)
                    return df
                except Exception as e:
                    self.loginfo("Error in retrying connecting to mysql server " + str(e), 'error')
            return None

    def update_database(self, sql, **kwargs):
        try:
            self.close_connection()
        except Exception as e_:
            self.loginfo("Connection reestablished")
        self.__init_db()
        conn = self.db_write
        cursor = conn.cursor()
        self.loginfo(message=sql)
        try:
            start_time = time.time()
            rows_affected = cursor.execute(sql)
            cursor.fetchall()
            conn.commit()
            time_diff = time.time() - start_time
            self.loginfo("Update Success , timeTaken = " + str(time_diff) + " rows=" + str(rows_affected))
            return rows_affected
        except Exception as e:
            self.loginfo("Error in executing sql: {}".format(sql), 'error')
            self.loginfo('Error details: {}'.format(e), 'error')
            if "Lost connection to MySQL server during query" in str(e) or \
                    "MySQL server has gone away" in str(e) or "(0, '')" in str(e) or "(2013, '')" in str(e) or "(2006, '')" in str(e):
                try:
                    try:
                        self.close_connection()
                    except Exception as e_:
                        pass
                    self.__init_db()
                    cursor = self.db_write.cursor()
                    rows_affected = cursor.execute(sql)
                    cursor.fetchall()
                    self.db_write.commit()
                    self.loginfo("Update Success " + str(rows_affected), 'debug')
                    return rows_affected
                except Exception as e:
                    self.loginfo("Error in retrying connecting to mysql server " + str(e), 'error')
            return False

    def update_database_and_return_id(self, sql):
        try:
            self.close_connection()
        except Exception as e_:
            self.loginfo("Connection reestablished")
        self.__init_db()
        self.loginfo("DB query: " + str(datetime.now()) + "Read Query: " + sql)
        cursor = self.db_write.cursor()
        self.loginfo("Update Query: " + sql, 'debug')
        try:
            start_time = time.time()
            cursor.execute(sql)
            cursor.fetchall()
            self.db_write.commit()
            time_diff = time.time() - start_time
            self.loginfo("Update Success , timeTaken: " + str(time_diff) + ', ID: ' + str(cursor.lastrowid), 'debug')
            return cursor.lastrowid
        except Exception as e:
            self.loginfo("Error in executing sql: {}".format(sql), 'error')
            self.loginfo('Error details: {}'.format(e), 'error')
            if "Lost connection to MySQL server during query" in str(e) or \
                    "MySQL server has gone away" in str(e) or "(0, '')" in str(e) or "(2013, '')" in str(e) or "(2006, '')" in str(e):
                try:
                    try:
                        self.close_connection()
                    except Exception as e_:
                        pass
                    self.__init_db()
                    cursor = self.db_write.cursor()
                    cursor.execute(sql)
                    cursor.fetchall()
                    self.db_write.commit()
                    self.loginfo("Update Success", 'debug')
                    return cursor.lastrowid
                except Exception as e:
                    self.loginfo("Error in retrying connecting to mysql server " + str(e), 'error')
            return None

    def close_connection(self):
        try:
            self.db_write.close()
        except Exception as e:
            self.loginfo("exception in closing connection :" + str(e))

    def loginfo(self, message, level='info'):
        if self.tag != '':
            message = '{}: {}'.format(self.tag, message)
        self.logger.log(message, level)

    def get_rabbitmq_consumer(self, exchange, queue, routing_keys, exchange_type='topic', durable='False'):
        username = self.config_json[self.database + '_rabbitmq']['rabbitmq_username']
        password = self.config_json[self.database + '_rabbitmq']['rabbitmq_password']
        host = self.config_json[self.database + '_rabbitmq']['rabbitmq_host']
        port = self.config_json[self.database + '_rabbitmq']['rabbitmq_port']
        virtual_host = self.config_json[self.database + '_rabbitmq']['rabbitmq_vhost']
        socket_timeout = self.config_json[self.database + '_rabbitmq']['socket_timeout']
        self.consumer_rabbitmq_creds = pika.PlainCredentials(username=username, password=password)
        self.consumer_connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, virtual_host=virtual_host, credentials=self.consumer_rabbitmq_creds, socket_timeout=socket_timeout, blocked_connection_timeout=60))
        self.consumer_channel = self.consumer_connection.channel()
        self.consumer_channel.exchange_declare(exchange=exchange, exchange_type=exchange_type, durable=durable)
        self.consumer_channel.queue_declare(queue=queue, durable=True)
        for routing_key in routing_keys:
            self.consumer_channel.queue_bind(exchange=exchange, queue=queue, routing_key=routing_key)

    def get_rabbitmq_publisher(self):
        try:
            username = self.config_json[self.database + '_rabbitmq']['rabbitmq_username']
            password = self.config_json[self.database + '_rabbitmq']['rabbitmq_password']
            host = self.config_json[self.database + '_rabbitmq']['rabbitmq_host']
            port = self.config_json[self.database + '_rabbitmq']['rabbitmq_port']
            virtual_host = self.config_json[self.database + '_rabbitmq']['rabbitmq_vhost']
            socket_timeout = self.config_json[self.database + '_rabbitmq']['socket_timeout']
            self.publisher_rabbitmq_creds = pika.PlainCredentials(username=username, password=password)
            self.publisher_connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, virtual_host=virtual_host, credentials=self.publisher_rabbitmq_creds, socket_timeout=socket_timeout, blocked_connection_timeout=60))
            self.publisher_channel = self.publisher_connection.channel()
            self.publisher_channel.exchange_declare(exchange=self.rabbitmq_publisher.get("exchange"), exchange_type=self.rabbitmq_publisher.get("exchange_type"), durable=self.rabbitmq_publisher.get("durable"))
        except Exception as e_:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.loginfo("except in publish " + str(e_) + ' ' + str(exc_tb.tb_lineno))

    def publish(self, event, routing_key=None):
        try:
            if not routing_key:
                routing_key = self.rabbitmq_publisher.get('routing_key')
            try:
                self.publisher_channel.basic_publish(exchange=self.rabbitmq_publisher.get("exchange"), routing_key=routing_key, body=event)
                self.loginfo("published : " + str(event))
            except Exception as e:
                try:
                    self.publisher_channel.close()
                except:
                    pass
                self.loginfo("exception >> " + str(e))
                username = self.config_json[self.database + '_rabbitmq']['rabbitmq_username']
                password = self.config_json[self.database + '_rabbitmq']['rabbitmq_password']
                host = self.config_json[self.database + '_rabbitmq']['rabbitmq_host']
                port = self.config_json[self.database + '_rabbitmq']['rabbitmq_port']
                virtual_host = self.config_json[self.database + '_rabbitmq']['rabbitmq_vhost']
                socket_timeout = self.config_json[self.database + '_rabbitmq']['socket_timeout']
                self.publisher_rabbitmq_creds = pika.PlainCredentials(username=username, password=password)
                self.publisher_connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, virtual_host=virtual_host, credentials=self.publisher_rabbitmq_creds, socket_timeout=socket_timeout, blocked_connection_timeout=60))
                self.publisher_channel = self.publisher_connection.channel()
                self.publisher_channel.exchange_declare(exchange=self.rabbitmq_publisher.get("exchange"), exchange_type=self.rabbitmq_publisher.get("exchange_type"), durable=self.rabbitmq_publisher.get("durable"))
                try:
                    self.publisher_channel.basic_publish(exchange=self.rabbitmq_publisher.get("exchange"), routing_key=routing_key, body=event)
                    self.loginfo("published : " + str(event))
                except Exception as e:
                    self.loginfo("except in inner publish : " + str(e))
        except Exception as e_:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.loginfo("except in publish " + str(e_) + ' ' + str(exc_tb.tb_lineno))
