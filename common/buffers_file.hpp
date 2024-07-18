// Copyright 2024 The Vulkan Shader Profiler authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <map>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

namespace vksp {

using buffer_map_key = std::pair<uint32_t, uint32_t>;
using buffer_map_val = std::pair<uint32_t, void *>;
using buffers_map = std::map<buffer_map_key, buffer_map_val>;

class BuffersFile {
public:
    BuffersFile(uint32_t dispatchId)
        : BuffersFile(dispatchId, true)
    {
    }
    BuffersFile(uint32_t dispatchId, bool oneFile)
        : m_version(1)
        , m_magic(0x766B7370) // VKSP in ASCII
        , m_dispatchId(dispatchId)
        , m_oneFile(oneFile)
    {
    }

    bool ReadFromFile(const char *filename)
    {
        FILE *fd = fopen(filename, "r");
        uint32_t file_header[3];
        if (fread(file_header, sizeof(file_header), 1, fd) != 1) {
            return false;
        }
        if (file_header[0] != m_magic || file_header[1] != m_version || file_header[2] != m_dispatchId) {
            return false;
        }
        while (true) {
            uint32_t buffer_header[3];
            if (fread(buffer_header, sizeof(buffer_header), 1, fd) != 1) {
                return false;
            }
            if (buffer_header[0] == UINT32_MAX || buffer_header[1] == UINT32_MAX || buffer_header[2] == UINT32_MAX) {
                break;
            }
            buffer_map_key key = std::make_pair(buffer_header[0], buffer_header[1]);

            uint32_t size = buffer_header[2];
            void *data = malloc(size);
            if (data == nullptr) {
                return false;
            }
            size_t byte_read = 0;
            while (byte_read != size) {
                byte_read += fread(&(((char *)data)[byte_read]), sizeof(char), size - byte_read, fd);
            }
            buffer_map_val val = std::make_pair(size, data);
            m_buffers[key] = val;
        }

        fclose(fd);
        return true;
    }

private:
    bool WriteToMultipleFiles(const char *filename)
    {
        for (auto &buffer : m_buffers) {
            std::string filename_str(filename);
            uint32_t set = buffer.first.first;
            uint32_t binding = buffer.first.second;
            uint32_t size = buffer.second.first;
            void *data = buffer.second.second;

            filename_str += "." + std::to_string(set) + "." + std::to_string(binding);
            FILE *fd = fopen(filename_str.c_str(), "w");
            if (fd == nullptr) {
                return false;
            }

            uint32_t byte_written = 0;
            while (byte_written != size) {
                byte_written += fwrite(&(((char *)data)[byte_written]), sizeof(char), size - byte_written, fd);
            }

            fclose(fd);
        }
        return true;
    }

    bool WriteToOneFile(const char *filename)
    {
        FILE *fd = fopen(filename, "w");
        if (fd == nullptr) {
            return false;
        }
        if (fwrite(&m_magic, sizeof(m_magic), 1, fd) != 1) {
            return false;
        }
        if (fwrite(&m_version, sizeof(m_version), 1, fd) != 1) {
            return false;
        }
        if (fwrite(&m_dispatchId, sizeof(m_dispatchId), 1, fd) != 1) {
            return false;
        }
        for (auto &buffer : m_buffers) {
            uint32_t set = buffer.first.first;
            uint32_t binding = buffer.first.second;
            uint32_t size = buffer.second.first;
            void *data = buffer.second.second;
            if (fwrite(&set, sizeof(set), 1, fd) != 1) {
                return false;
            }
            if (fwrite(&binding, sizeof(binding), 1, fd) != 1) {
                return false;
            }
            if (fwrite(&size, sizeof(size), 1, fd) != 1) {
                return false;
            }
            uint32_t byte_written = 0;
            while (byte_written != size) {
                byte_written += fwrite(&(((char *)data)[byte_written]), sizeof(char), size - byte_written, fd);
            }
        }
        uint32_t eof[3] = { UINT32_MAX, UINT32_MAX, UINT32_MAX };
        if (fwrite(eof, sizeof(eof), 1, fd) != 1) {
            return false;
        }
        fclose(fd);

        return true;
    }

public:
    bool WriteToFile(const char *filename)
    {
        if (m_oneFile) {
            return WriteToOneFile(filename);
        } else {
            return WriteToMultipleFiles(filename);
        }
    }

    void AddBuffer(uint32_t set, uint32_t binding, uint32_t size, void *data)
    {
        buffer_map_key key = std::make_pair(set, binding);
        buffer_map_val val = std::make_pair(size, data);
        m_buffers[key] = val;
    }

    buffers_map *GetBuffers() { return &m_buffers; }

private:
    const uint32_t m_version;
    const uint32_t m_magic;
    const uint32_t m_dispatchId;
    buffers_map m_buffers;
    bool m_oneFile;
};
}
