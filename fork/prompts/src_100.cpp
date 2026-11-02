// file: common/json-schema-to-grammar.cpp
#include "json-schema-to-grammar.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include <algorithm>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = common_json;

static std::string build_repetition(const std::string & item_rule, int min_items, int max_items, 