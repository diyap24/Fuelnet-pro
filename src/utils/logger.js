'use strict';

const { createLogger, format, transports } = require('winston');
const path = require('path');
const fs = require('fs');

const logDir = path.resolve(process.env.LOG_FILE ? path.dirname(process.env.LOG_FILE) : './logs');
if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, { recursive: true });

const logger = createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: format.combine(
    format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
    format.errors({ stack: true }),
    format.splat(),
    format.json()
  ),
  defaultMeta: { service: 'fuelnet-pro' },
  transports: [
    new transports.Console({
      format: format.combine(
        format.colorize(),
        format.printf(({ timestamp, level, message, ...meta }) => {
          const extras = Object.keys(meta).length ? ' ' + JSON.stringify(meta) : '';
          return `[${timestamp}] ${level}: ${message}${extras}`;
        })
      )
    }),
    new transports.File({
      filename: process.env.LOG_FILE || './logs/fuelnet.log',
      maxsize: 5 * 1024 * 1024, // 5MB
      maxFiles: 5,
      tailable: true
    })
  ]
});

module.exports = logger;
