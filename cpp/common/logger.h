/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "stringUtils.h"
#include <NvInferRuntime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace trt_edgellm
{

namespace logger
{

/*!
 * @brief Source code location information for automatic tracking
 *
 * Captures file, function, and line number information for logging.
 */
struct SourceLocation
{
    char const* file;     //!< Source file path
    char const* function; //!< Function name
    int32_t lineNumber;   //!< Line number

    /*!
     * @brief Constructor with manual location capture
     * @param f Source file path
     * @param func Function name
     * @param l Line number
     */
    SourceLocation(char const* f, char const* func, int32_t l)
        : file(f)
        , function(func)
        , lineNumber(l)
    {
    }
};

/**
 * @brief Enhanced Logger with automatic location tracking and nvinfer1 compatibility
 *
 * Features:
 * - Automatic source location tracking (file:line:function)
 * - Configurable formatting (timestamps, location info)
 * - nvinfer1::ILogger interface for TensorRT integration
 * - Multiple log levels with performance optimizations
 */
class EdgeLLMLogger : public nvinfer1::ILogger
{
public:
    EdgeLLMLogger() = default;
    ~EdgeLLMLogger()
    {
        closeFileOutput();
    }

    /*!
     * @brief nvinfer1::ILogger interface implementation for TensorRT integration
     * @param severity Log severity level
     * @param msg Log message
     */
    void log(nvinfer1::ILogger::Severity severity, char const* msg) noexcept override
    {
        // Create source location for external library messages
        SourceLocation extLoc("TensorRT", "TensorRT_Internal", 0);
        logWithLocation(severity, msg, extLoc);
    }

    /*!
     * @brief Core logging function with automatic location tracking and formatting
     * @param level Log severity level
     * @param msg Log message
     * @param loc Source location information
     */
    void logWithLocation(nvinfer1::ILogger::Severity level, std::string const& msg, SourceLocation const& loc)
    {
        if (!shouldLog(level))
        {
            return;
        }

        std::string formattedMsg = formatLogEntry(level, msg, loc);
        std::lock_guard<std::mutex> lock(mSinkMutex);

        if (mConsoleEnabled)
        {
            std::ostream& stream = (level <= nvinfer1::ILogger::Severity::kWARNING) ? std::cerr : std::cout;
            stream << formattedMsg << std::endl;
        }

        if (mFileEnabled)
        {
            if (!ensureFileSinkOpen())
            {
                reportFileSinkFailureOnce("failed to open log file sink");
                return;
            }

            if (shouldRotate(formattedMsg.size() + 1U))
            {
                rotateLogFile();
            }

            mLogFile << formattedMsg << '\n';
            mLogFile.flush();
            if (!mLogFile.good())
            {
                reportFileSinkFailureOnce("failed to write log file");
            }
        }
    }

    /*!
     * @brief Log debug message with location tracking
     * @param msg Log message
     * @param loc Source location information
     */
    void debug(std::string const& msg, SourceLocation const& loc)
    {
        logWithLocation(nvinfer1::ILogger::Severity::kVERBOSE, msg, loc);
    }

    /*!
     * @brief Log info message with location tracking
     * @param msg Log message
     * @param loc Source location information
     */
    void info(std::string const& msg, SourceLocation const& loc)
    {
        logWithLocation(nvinfer1::ILogger::Severity::kINFO, msg, loc);
    }

    /*!
     * @brief Log warning message with location tracking
     * @param msg Log message
     * @param loc Source location information
     */
    void warning(std::string const& msg, SourceLocation const& loc)
    {
        logWithLocation(nvinfer1::ILogger::Severity::kWARNING, msg, loc);
    }

    /*!
     * @brief Log error message with location tracking
     * @param msg Log message
     * @param loc Source location information
     */
    void error(std::string const& msg, SourceLocation const& loc)
    {
        logWithLocation(nvinfer1::ILogger::Severity::kERROR, msg, loc);
    }

    /*!
     * @brief Set minimum logging level
     * @param level Minimum severity level to log
     */
    void setLevel(nvinfer1::ILogger::Severity level)
    {
        mMinLevel = level;
    }

    /*!
     * @brief Get current logging level
     * @return Current minimum severity level
     */
    nvinfer1::ILogger::Severity getLevel() const
    {
        return mMinLevel;
    }

    /*!
     * @brief Configure whether to show timestamps in log output
     * @param show true to show timestamps, false to hide
     */
    void setShowTimestamp(bool show)
    {
        mShowTimestamp = show;
    }

    /*!
     * @brief Configure whether to show location info in log output
     * @param show true to show location, false to hide
     */
    void setShowLocation(bool show)
    {
        mShowLocation = show;
    }

    /*!
     * @brief Configure whether to show function names in log output
     * @param show true to show function names, false to hide
     */
    void setShowFunction(bool show)
    {
        mShowFunction = show;
    }

    /*!
     * @brief Configure whether to keep writing logs to stdout/stderr.
     * @param enabled true to keep console output, false to disable console output.
     */
    void setConsoleEnabled(bool enabled)
    {
        std::lock_guard<std::mutex> lock(mSinkMutex);
        mConsoleEnabled = enabled;
    }

    /*!
     * @brief Enable file logging with rotation and retention cleanup.
     * @param logDir Target log directory.
     * @param baseName Base name of log files.
     * @param maxFileBytes Rotate when active log reaches this size.
     * @param maxFiles Maximum number of rotated files to keep (0 means unlimited).
     * @param retentionDays Delete rotated logs older than this many days (0 means unlimited).
     * @return true when the file sink is ready, false when initialization failed.
     */
    bool setFileOutput(std::string const& logDir, std::string const& baseName, size_t maxFileBytes, int32_t maxFiles,
        int32_t retentionDays)
    {
        std::lock_guard<std::mutex> lock(mSinkMutex);
        closeFileOutputLocked();

        mLogDir = std::filesystem::path(logDir);
        mBaseName = baseName.empty() ? std::string("edgellm") : baseName;
        mMaxFileBytes = (maxFileBytes == 0) ? (100U * 1024U * 1024U) : maxFileBytes;
        mMaxFiles = std::max<int32_t>(0, maxFiles);
        mRetentionDays = std::max<int32_t>(0, retentionDays);
        mActiveLogPath = mLogDir / (mBaseName + ".log");
        mFileEnabled = true;
        mFileErrorReported = false;

        std::error_code ec;
        std::filesystem::create_directories(mLogDir, ec);
        if (ec)
        {
            mFileEnabled = false;
            return false;
        }

        cleanupOldLogFiles();
        if (!ensureFileSinkOpen())
        {
            mFileEnabled = false;
            return false;
        }
        return true;
    }

    /*!
     * @brief Disable file logging and close open log file handle.
     */
    void closeFileOutput()
    {
        std::lock_guard<std::mutex> lock(mSinkMutex);
        closeFileOutputLocked();
    }

private:
    nvinfer1::ILogger::Severity mMinLevel = nvinfer1::ILogger::Severity::kINFO;
    bool mShowTimestamp = true;
    bool mShowLocation = true;
    bool mShowFunction = true;
    bool mConsoleEnabled = true;
    bool mFileEnabled = false;
    bool mFileErrorReported = false;
    std::filesystem::path mLogDir;
    std::filesystem::path mActiveLogPath;
    std::string mBaseName{"edgellm"};
    size_t mMaxFileBytes = 100U * 1024U * 1024U;
    int32_t mMaxFiles = 20;
    int32_t mRetentionDays = 7;
    std::ofstream mLogFile;
    mutable std::mutex mSinkMutex;

    bool shouldLog(nvinfer1::ILogger::Severity level) const
    {
        return level <= mMinLevel; // Note: lower values are more severe in TensorRT
    }

    bool ensureFileSinkOpen()
    {
        if (mLogFile.is_open())
        {
            return true;
        }

        std::error_code ec;
        std::filesystem::create_directories(mLogDir, ec);
        if (ec)
        {
            return false;
        }
        mLogFile.open(mActiveLogPath, std::ios::app);
        return mLogFile.is_open();
    }

    bool shouldRotate(size_t incomingBytes) const
    {
        if (mMaxFileBytes == 0)
        {
            return false;
        }

        std::error_code ec;
        auto currentSize = std::filesystem::file_size(mActiveLogPath, ec);
        if (ec)
        {
            return false;
        }
        return currentSize + incomingBytes > mMaxFileBytes;
    }

    void rotateLogFile()
    {
        mLogFile.close();

        std::error_code ec;
        if (std::filesystem::exists(mActiveLogPath, ec))
        {
            auto rotatedPath = buildRotatedLogPath();
            std::filesystem::rename(mActiveLogPath, rotatedPath, ec);
            if (ec)
            {
                reportFileSinkFailureOnce("failed to rotate log file");
            }
        }

        cleanupOldLogFiles();
        if (!ensureFileSinkOpen())
        {
            reportFileSinkFailureOnce("failed to reopen log file after rotation");
        }
    }

    std::filesystem::path buildRotatedLogPath() const
    {
        auto now = std::chrono::system_clock::now();
        auto timeT = std::chrono::system_clock::to_time_t(now);
        std::tm localTm{};
#if defined(_WIN32)
        localtime_s(&localTm, &timeT);
#else
        localtime_r(&timeT, &localTm);
#endif
        std::ostringstream ts;
        ts << std::put_time(&localTm, "%Y%m%d-%H%M%S");

        for (int32_t seq = 0; seq < 1000; ++seq)
        {
            std::string filename = mBaseName + "." + ts.str() + "." + std::to_string(seq) + ".log";
            auto candidate = mLogDir / filename;
            std::error_code ec;
            if (!std::filesystem::exists(candidate, ec))
            {
                return candidate;
            }
        }
        return mLogDir / (mBaseName + "." + ts.str() + ".overflow.log");
    }

    void cleanupOldLogFiles()
    {
        std::error_code ec;
        if (!std::filesystem::exists(mLogDir, ec))
        {
            return;
        }

        std::vector<std::filesystem::path> rotatedLogs;
        auto const prefix = mBaseName + ".";

        for (auto const& entry : std::filesystem::directory_iterator(mLogDir, ec))
        {
            if (ec || !entry.is_regular_file())
            {
                continue;
            }
            auto const filename = entry.path().filename().string();
            if (filename.rfind(prefix, 0) != 0 || entry.path().extension() != ".log")
            {
                continue;
            }
            rotatedLogs.push_back(entry.path());
        }

        if (mRetentionDays > 0)
        {
            auto const cutoff = std::filesystem::file_time_type::clock::now()
                - std::chrono::hours(static_cast<int64_t>(mRetentionDays) * 24);
            for (auto it = rotatedLogs.begin(); it != rotatedLogs.end();)
            {
                auto const writeTime = std::filesystem::last_write_time(*it, ec);
                if (!ec && writeTime < cutoff)
                {
                    std::filesystem::remove(*it, ec);
                    it = rotatedLogs.erase(it);
                    continue;
                }
                ec.clear();
                ++it;
            }
        }

        if (mMaxFiles > 0 && static_cast<int32_t>(rotatedLogs.size()) > mMaxFiles)
        {
            std::sort(rotatedLogs.begin(), rotatedLogs.end(),
                [](std::filesystem::path const& a, std::filesystem::path const& b) {
                    std::error_code leftEc;
                    std::error_code rightEc;
                    auto const leftTime = std::filesystem::last_write_time(a, leftEc);
                    auto const rightTime = std::filesystem::last_write_time(b, rightEc);
                    if (leftEc || rightEc)
                    {
                        return a.string() < b.string();
                    }
                    return leftTime < rightTime;
                });

            auto const deleteCount = static_cast<size_t>(rotatedLogs.size() - mMaxFiles);
            for (size_t i = 0; i < deleteCount; ++i)
            {
                std::filesystem::remove(rotatedLogs[i], ec);
                ec.clear();
            }
        }
    }

    void reportFileSinkFailureOnce(char const* message)
    {
        if (mFileErrorReported)
        {
            return;
        }
        mFileErrorReported = true;
        std::cerr << "[LOGGER] " << message << std::endl;
    }

    void closeFileOutputLocked()
    {
        if (mLogFile.is_open())
        {
            mLogFile.close();
        }
        mFileEnabled = false;
        mFileErrorReported = false;
    }

    std::string formatLogEntry(
        nvinfer1::ILogger::Severity level, std::string const& msg, SourceLocation const& loc) const
    {
        std::ostringstream oss;

        // Timestamp
        if (mShowTimestamp)
        {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

            oss << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "." << std::setfill('0') << std::setw(3)
                << ms.count() << "] ";
        }

        // Log level
        oss << "[" << getLevelString(level) << "]";

        // Location information
        if (mShowLocation && loc.file)
        {
            // Check if this is TensorRT message
            if (std::string(loc.file) == "TensorRT")
            {
                oss << " [TensorRT]";
            }
            else
            {
                std::filesystem::path p(loc.file);
                oss << " [" << p.filename().string() << ":" << loc.lineNumber;
                if (mShowFunction && loc.function)
                {
                    oss << ":" << loc.function;
                }
                oss << "]";
            }
        }

        // Message
        oss << " " << msg;

        return oss.str();
    }

    char const* getLevelString(nvinfer1::ILogger::Severity level) const
    {
        switch (level)
        {
        case nvinfer1::ILogger::Severity::kVERBOSE: return "DEBUG";
        case nvinfer1::ILogger::Severity::kINFO: return "INFO";
        case nvinfer1::ILogger::Severity::kWARNING: return "WARNING";
        case nvinfer1::ILogger::Severity::kERROR: return "ERROR";
        default: return "UNKNOWN";
        }
    }
};

/*!
 * @brief RAII-based function tracer for automatic entry/exit logging
 *
 * Creates automatic log messages when entering and exiting a scope.
 * Useful for tracing function execution flow.
 */
class ScopedFunctionTracer
{
public:
    /*!
     * @brief Constructor that logs function entry
     * @param logger Logger instance to use
     * @param funcName Name of the function being traced
     * @param loc Source location information
     */
    ScopedFunctionTracer(EdgeLLMLogger& logger, char const* funcName, SourceLocation const& loc)
        : mLogger(logger)
        , mFuncName(funcName)
        , mLoc(loc)
    {
        mLogger.debug("-> Entering " + mFuncName, mLoc);
    }

    /*!
     * @brief Destructor that logs function exit
     */
    ~ScopedFunctionTracer()
    {
        mLogger.debug("<- Exiting " + mFuncName, mLoc);
    }

private:
    EdgeLLMLogger& mLogger;
    std::string mFuncName;
    SourceLocation mLoc;
};

} // namespace logger

inline logger::EdgeLLMLogger gLogger{};

// Primary logging macros with automatic location tracking
// Usage: LOG_DEBUG("Value: %d", value); LOG_INFO("Message: %s", msg);

#define LOG_DEBUG(...)                                                                                                 \
    gLogger.debug(format::fmtstr(__VA_ARGS__), trt_edgellm::logger::SourceLocation(__FILE__, __FUNCTION__, __LINE__))

#define LOG_INFO(...)                                                                                                  \
    gLogger.info(format::fmtstr(__VA_ARGS__), trt_edgellm::logger::SourceLocation(__FILE__, __FUNCTION__, __LINE__))

#define LOG_WARNING(...)                                                                                               \
    gLogger.warning(format::fmtstr(__VA_ARGS__), trt_edgellm::logger::SourceLocation(__FILE__, __FUNCTION__, __LINE__))

#define LOG_ERROR(...)                                                                                                 \
    gLogger.error(format::fmtstr(__VA_ARGS__), trt_edgellm::logger::SourceLocation(__FILE__, __FUNCTION__, __LINE__))

// Conditional logging macros for performance-critical code
#define LOG_DEBUG_IF(condition, ...)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (condition)                                                                                                 \
        {                                                                                                              \
            LOG_DEBUG(__VA_ARGS__);                                                                                    \
        }                                                                                                              \
    } while (0)

#define LOG_INFO_IF(condition, ...)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (condition)                                                                                                 \
        {                                                                                                              \
            LOG_INFO(__VA_ARGS__);                                                                                     \
        }                                                                                                              \
    } while (0)

#define LOG_WARNING_IF(condition, ...)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        if (condition)                                                                                                 \
        {                                                                                                              \
            LOG_WARNING(__VA_ARGS__);                                                                                  \
        }                                                                                                              \
    } while (0)

#define LOG_ERROR_IF(condition, ...)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (condition)                                                                                                 \
        {                                                                                                              \
            LOG_ERROR(__VA_ARGS__);                                                                                    \
        }                                                                                                              \
    } while (0)

// Function tracing macro for automatic entry/exit logging
#define LOG_TRACE_FUNCTION()                                                                                           \
    trt_edgellm::logger::ScopedFunctionTracer gTracer(                                                                 \
        gLogger, __FUNCTION__, trt_edgellm::logger::SourceLocation(__FILE__, __FUNCTION__, __LINE__))

} // namespace trt_edgellm
