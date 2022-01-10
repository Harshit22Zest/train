import gzip
import json
import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler

from typing import Optional


class Logger(object):
    """
    A thread safe class for logging
    """

    def __init__(self, debug=0, mode=0, log_file=None, level=logging.INFO, config_file_path='config.json'):
        # type: (int, int, Optional[str], int, Optional[str]) -> None
        """
        :param debug: Wether debugging enabled or not
        :param mode: Logs to console if 0, and to file if 1
        :param log_file: File to write logs to
        :param level: Logging level to show (Default: info)
        """
        if debug and mode and not log_file:
            raise ValueError('logFile CANNOT be None if debug = 1 and mode = 1')

        self.debug = debug
        self.mode = mode
        self.__log_file = log_file
        self.__config_file_path = config_file_path

        if log_file:
            config_json = json.loads(open(self.__config_file_path).read())
            config_json = config_json.get(config_json.get("env"))
            logs_dir = config_json["_logs"]
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            log_file = os.path.join(logs_dir, log_file)

            logging.getLogger("pika").setLevel(logging.WARNING)
            fmt = '%(asctime)s.%(msecs)03d: %(threadName)s: %(levelname)s: %(message)s'
            formatter = logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')
            self.__logger = logging.getLogger(log_file)
            self.__logger.setLevel(level)
            handler = CustomRotatingFileHandler(filename=log_file)
            handler.setFormatter(formatter)
            if not self.__logger.handlers:
                self.__logger.addHandler(handler)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if {'debug', 'mode', '__log_file'}.issubset(self.__dict__) and self.debug and self.mode and not self.__log_file:
            raise ValueError('logFile CANNOT be None if debug = 1 and mode = 1')

    def log(self, msg, lvl='info'):
        """
        :param msg: log message
        :param lvl: 'debug', 'info', 'warn', 'error', 'critical' (deafult: 'info')
        """
        if self.debug == 1:
            if self.mode:
                if lvl == 'debug':
                    self.__logger.debug(msg)
                elif lvl == 'info':
                    self.__logger.info(msg)
                elif lvl == 'warn':
                    self.__logger.warning(msg)
                elif lvl == 'error':
                    self.__logger.error(msg)
                elif lvl == 'critical':
                    self.__logger.critical(msg)
            else:
                # print(msg)
                pass


class CustomRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, when='midnight', backup_count=15, max_bytes=1073741824):  # 1 GB
        """
        This is just a combination of TimedRotatingFileHandler and RotatingFileHandler
        (adds maxBytes to TimedRotatingFileHandler)
        """
        TimedRotatingFileHandler.__init__(self, filename=filename, when=when, backupCount=backup_count)
        self.maxBytes = max_bytes

    @staticmethod
    def doArchive(old_log):
        with open(old_log, 'rb') as log:
            with gzip.open(old_log + '.gz', 'wb') as comp_log:
                comp_log.writelines(log)
        os.remove(old_log)

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.

        Basically, see if the supplied record would cause the file to exceed the size limit we have.

        we are also comparing times
        """
        if self.stream is None:  # delay was set...
            self.stream = self._open()
        if self.maxBytes > 0:  # are we rolling over?
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)  # due to non-posix-compliant Windows feature
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return 1
        t = int(time.time())
        if t >= self.rolloverAt:
            return 1
        return 0

    def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.
        """
        if self.stream:
            self.stream.close()
        # get the time that this sequence started at and make it a TimeTuple
        current_time = int(time.time())
        dst_now = time.localtime(current_time)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            time_tuple = time.gmtime(t)
        else:
            time_tuple = time.localtime(t)
            dst_then = time_tuple[-1]
            if dst_now != dst_then:
                if dst_now:
                    addend = 3600
                else:
                    addend = -3600
                time_tuple = time.localtime(t + addend)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, time_tuple)
        if self.backupCount > 0:
            cnt = 1
            dfn2 = "%s.%02d" % (dfn, cnt)
            while os.path.exists(dfn2 + '.gz'):
                cnt += 1
                dfn2 = "%s.%02d" % (dfn, cnt)
            os.rename(self.baseFilename, dfn2)
            self.doArchive(dfn2)
            for s in self.getFilesToDelete():
                os.remove(s)
        else:
            if os.path.exists(dfn):
                os.remove(dfn)
            os.rename(self.baseFilename, dfn)
            self.doArchive(dfn)
        self.mode = 'w'
        self.stream = self._open()
        new_rollover_at = self.computeRollover(current_time)
        while new_rollover_at <= current_time:
            new_rollover_at = new_rollover_at + self.interval
        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dst_at_rollover = time.localtime(new_rollover_at)[-1]
            if dst_now != dst_at_rollover:
                if not dst_now:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:  # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                new_rollover_at += addend
        self.rolloverAt = new_rollover_at

    def getFilesToDelete(self):
        """
        Determine the files to delete when rolling over.

        More specific than the earlier method, which just used glob.glob().
        """
        dir_name, base_name = os.path.split(self.baseFilename)
        file_names = os.listdir(dir_name)
        result = []
        prefix = base_name + "."
        plen = len(prefix)
        for file_name in file_names:
            if file_name[:plen] == prefix:
                suffix = file_name[plen:-6]
                if self.extMatch.match(suffix):
                    result.append(os.path.join(dir_name, file_name))
        result.sort()
        if len(result) < self.backupCount:
            result = []
        else:
            result = result[:len(result) - self.backupCount]
        return result
