#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>
#include <unordered_map>

struct common_params;

// generator-like API for HTTP response generation
// this object response with one of the 2 modes:
// 1) normal response: `data` contains the full response body
// 2) streaming response: each call to next(output) generates the next chunk
//    when next(output) returns false, no more data after the current chunk
//    note: some chunks can be empty, in which case no data is sent for that chunk
struct server_http_res {
    std::string content_type = "application/json; charset=utf-8";
    int status = 200;
    std::string data;
    std::map<std::string, std::string> headers;

    std::function<bool(std::string &)> next = nullptr;
    bool is_stream() const {
        return next != nullptr;
    }

    // fired before req and res are destroyed
    virtual void on_complete() {}

    virtual ~server_http_res() = default;
};

// unique pointer, used by set_chunked_content_provider
// httplib requires the stream provider to be stored in heap
using server_http_res_ptr = std::unique_ptr<server_http_res>;
using raw_buffer = std::vector<uint8_t>;

struct uploaded_file {
    raw_buffer data;
    std::string filename;
    std::string content_type;
};

struct server_http_req {
    std::map<std::string, std::string> params; // path_params + query_params
    std::map<std::string, std::string> headers; // used by MCP proxy
    std::string path;
    std::string query_string; // query parameters string (e.g. "action=save")
    std::string body;
    std::map<std::string, uploaded_file> files; // used for file uploads (form data)
    const std::function<bool()> & should_stop;

    std::string get_param(const std::string & key, const std::string & def = "") const {
        auto it = params.find(key);
        if (it != params.end()) {
            return it->second;
        }
        return def;
    }
};

// fork: live state for GET /status, written by the server loop and the loader, read by any thread, also while the model loads (the engine reports, run.py collects)
struct server_live_status {
    enum { LOADING = 0, IDLE = 1, PREFILL = 2, GEN = 3 };
    std::atomic<int>     phase{LOADING};
    std::atomic<float>   load_progress{0.0f};   // 0..1 of the stage named below
    std::atomic<int64_t> t_start_us{0};         // request start (prompt processing begins)
    std::atomic<int64_t> t_prompt_last_us{0};   // end of the last finished prompt batch; generation counts from here
    std::atomic<int64_t> t_gen_last_us{0};      // last generated token
    std::atomic<int64_t> n_prompt{0};           // prompt tokens finished
    std::atomic<int64_t> n_prompt_total{0};     // prompt tokens of the request
    std::atomic<int64_t> n_gen{0};
    std::atomic<int64_t> n_requests{0};         // requests finished since the start
    std::mutex           m;
    std::string          stage;                 // the load stage (text_model, ...) while loading
};
extern server_live_status g_live_status;

struct server_http_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::atomic<bool> is_ready = false;

    // note: the handler should never throw exceptions
    using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;
    mutable std::unordered_map<std::string, handler_t> handlers;

    // websocket connection, see ws()
    struct ws_conn {
        // blocking read, returns 1 for a text message, 2 for a binary message, 0 when the connection is closed
        virtual int  read(std::string & msg) = 0;
        // thread-safe, can be called while another thread is in read()
        virtual bool send(const std::string & msg) = 0;
        virtual bool send_binary(const std::string & msg) = 0;
        virtual bool is_open() const = 0;
        virtual void close() = 0;
        virtual ~ws_conn() = default;
    };
    // the handler runs on its own HTTP thread for the whole lifetime of the connection
    using ws_handler_t = std::function<void(const server_http_req & req, ws_conn & conn)>;

    std::string path_prefix;
    int port    = 9931;
    bool is_ssl = false;

    server_http_context();
    ~server_http_context();

    bool init(const common_params & params);
    bool start();
    void stop() const;
    void join();

    void get(const std::string & path, const handler_t & handler) const;
    void post(const std::string & path, const handler_t & handler) const;
    void del(const std::string & path, const handler_t & handler) const;
    void ws(const std::string & path, const ws_handler_t & handler) const;

    // Register the Google Cloud Platform (Vertex AI) compat (AIP_PREDICT_ROUTE env var, or /predict)
    // Must be called AFTER all other API routes are registered
    void register_gcp_compat() const;

    // for debugging
    std::vector<std::string> listening_addresses;

private:
    bool init_listener(const common_params & params);
};
