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

static std::string build_repetition(const std::string & item_rule, int min_items, int max_items, const std::string & separator_rule = "") {
    auto has_max = max_items != std::numeric_limits<int>::max();

    if (max_items == 0) {
        return "";
    }
    if (min_items == 0 && max_items == 1) {
        return item_rule + "?";
    }

    if (separator_rule.empty()) {
        if (min_items == 1 && !has_max) {
            return item_rule + "+";
        }
        if (min_items == 0 && !has_max) {
            return item_rule + "*";
        }
        return item_rule + "{" + std::to_string(min_items) + "," + (has_max ? std::to_string(max_items) : "") + "}";
    }

    auto result = item_rule + " " + build_repetition("(" + separator_rule + " " + item_rule + ")", min_items == 0 ? 0 : min_items - 1, has_max ? max_items - 1 : max_items);
    if (min_items == 0) {
        result = "(" + result + ")?";
    }
    return result;
}

static void build_min_max_int(int64_t min_value, int64_t max_value, std::stringstream & out, int decimals_left = 16, bool top_level = true) {
    auto has_min = min_value != std::numeric_limits<int64_t>::min();
    auto has_max = max_value != std::numeric_limits<int64_t>::max();

    auto digit_range = [&](char from, char to) {
        out << "[";
        if (from == to) {
            out << from;
        } else {
            out << from << "-" << to;
        }
        out << "]";
    };
    auto more_digits = [&](int min_digits, int max_digits) {
        out << "[0-9]";
        if (min_digits == max_digits && min_digits == 1) {
            return;
        }
        out << "{";
        out << min_digits;
        if (max_digits != min_digits) {
            out << ",";
            if (max_digits != std::numeric_limits<int>::max()) {
                out << max_digits;
            }
        }
        out << "}";
    };
    std::function<void(const std::string_view &, const std::string_view &)> uniform_range =
        [&](const std::string_view & from, const std::string_view & to) {
            size_t i = 0;
            while (i < from.length() && i < to.length() && from[i] == to[i]) {
                i++;
            }
            if (i > 0) {
                out << "\"" << from.substr(0, i) << "\"";
            }
            if (i < from.length() && i < to.length()) {
                if (i > 0) {
                    out << " ";
                }
                auto sub_len = from.length() - i - 1;
                if (sub_len > 0) {
                    auto from_sub = from.substr(i + 1);
                    auto to_sub = to.substr(i + 1);
                    auto sub_zeros = string_repeat("0", sub_len);
                    auto sub_nines = string_repeat("9", sub_len);

                    auto to_reached = false;
                    out << "(";
                    if (from_sub == sub_zeros) {
                        digit_range(from[i], to[i] - 1);
                        out << " ";
                        more_digits(sub_len, sub_len);
                    } else {
                        out << "[" << from[i] << "] ";
                        out << "(";
                        uniform_range(from_sub, sub_nines);
                        out << ")";
                        if (from[i] < to[i] - 1) {
                            out << " | ";
                            if (to_sub == sub_nines) {
                                digit_range(from[i] + 1, to[i]);
                                to_reached = true;
                            } else {
                                digit_range(from[i] + 1, to[i] - 1);
                            }
                            out << " ";
                            more_digits(sub_len, sub_len);
                        }
                    }
                    if (!to_reached) {
                        out << " | ";
                        digit_range(to[i], to[i]);
                        out << " ";
                        uniform_range(sub_zeros, to_sub);
                    }
                    out << ")";
                } else {
                    out << "[" << from[i] << "-" << to[i] << "]";
                }
            }
        };

    if (has_min && has_max) {
        if (min_value < 0 && max_value < 0) {
            out << "\"-\" (";
            build_min_max_int(-max_value, -min_value, out, decimals_left, /* top_level= */ true);
            out << ")";
            return;
        }

        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(0, -min_value, out, decimals_left, /* top_level= */ true);
            out << ") | ";
            min_value = 0;
        }

        auto min_s = std::to_string(min_value);
        auto max_s = std::to_string(max_value);
        auto min_digits = min_s.length();
        auto max_digits = max_s.length();

        for (auto digits = min_digits; digits < max_digits; digits++) {
            uniform_range(min_s, string_repeat("9", digits));
            min_s = "1" + string_repeat("0", digits);
            out << " | ";
        }
        uniform_range(min_s, max_s);
        return;
    }

    auto less_decimals = std::max(decimals_left - 1, 1);

    if (has_min) {
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(std::numeric_limits<int64_t>::min(), -min_value, out, decimals_left, /* top_level= */ false);
            out << ") | [0] | [1-9] ";
            more_digits(0, decimals_left - 1);
        } else if (min_value == 0) {
            if (top_level) {
                out << "[0] | [1-9] ";
                more_digits(0, less_decimals);
            } else {
                more_digits(1, decimals_left);
            }
        } else if (min_value <= 9) {
            char c = '0' + min_value;
            auto range_start = top_level ? '1' : '0';
            if (c > range_start) {
                digit_range(range_start, c - 1);
                out << " ";
                more_digits(1, less_decimals);
                out << " | ";
            }
            digit_range(c, '9');
            out << " ";
            more_digits(0, less_decimals);
        } else {
            auto min_s = std::to_string(min_value);
            auto len = min_s.length();
            auto c = min_s[0];

            if (c > '1') {
                digit_range(top_level ? '1' : '0', c - 1);
                out << " ";
                more_digits(len, less_decimals);
                out << " | ";
            }
            digit_range(c, c);
            out << " (";
            build_min_max_int(std::stoll(min_s.substr(1)), std::numeric_limits<int64_t>::max(), out, less_decimals, /* top_level= */ false);
            out << ")";
            if (c < '9') {
                out << " | ";
                digit_range(c + 1, '9');
                out << " ";
                more_digits(len - 1, less_decimals);
            }
        }
        return;
    }

    if (has_max) {
        if (max_value >= 0) {
            if (top_level) {
                out << "\"-\" [1-9] ";
                more_digits(0, less_decimals);
                out << " | ";
            }
            build_min_max_int(0, max_value, out, decimals_left, /* top_level= */ true);
        } else {
            out << "\"-\" (";
            build_min_max_int(-max_value, std::numeric_limits<int64_t>::max(), out, decimals_left, /* top_level= */ false);
            out << ")";
        }
        return;
    }

    throw std::runtime_error("At least one of min_value or max_value must be set");
}

const std::string SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
    std::string content;
    std::vector<std::string> deps;
};

static std::unordered_map<std::string, BuiltinRule> PRIMITIVE_RULES = {
    {"boolean", {"(\"true\" | \"false\")", {}}},
    {"decimal-part", {"[0-9]{1,16}", {}}},
    {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
    {"number", {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", {"integral-part", "decimal-part"}}},
    {"integer", {"(\"-\"? integral-part)", {"integral-part"}}},
    {"value", {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
    {"object", {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"", {"string", "value"}}},
    {"array", {"\"[\" space ( value (\",\" space value)* )? space \"]\"", {"value"}}},
    {"uuid", {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"", {}}},
    {"char",   {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {}}},
    {"string", {"\"\\\"\" char* \"\\\"\"", {"char"}}},
    {"null", {"\"null\"", {}}},
};

static std::unordered_map<std::string, BuiltinRule> STRING_FORMAT_RULES = {
    {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
    {"time", {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", {}}},
    {"date-time", {"date \"T\" time", {"date", "time"}}},
    {"date-string", {"\"\\\"\" date \"\\\"\"", {"date"}}},
    {"time-string", {"\"\\\"\" time \"\\\"\"", {"time"}}},
    {"date-time-string", {"\"\\\"\" date-time \"\\\"\"", {"date-time"}}}
};

static bool is_reserved_name(const std::string & name) {
    static const std::unordered_set<std::string> RESERVED_NAMES = [] {
        std::unordered_set<std::string> s;
        s.insert("root");
        for (const auto & p : PRIMITIVE_RULES) {
            s.insert(p.first);
        }
        for (const auto & p : STRING_FORMAT_RULES) {
            s.insert(p.first);
        }
        return s;
    }();
    return RESERVED_NAMES.find(name) != RESERVED_NAMES.end();
}

static std::regex INVALID_RULE_CHARS_RE("[^a-zA-Z0-9-]+");
static std::regex GRAMMAR_LITERAL_ESCAPE_RE("[\r\n\"\\\\]");
static std::regex GRAMMAR_RANGE_LITERAL_ESCAPE_RE("[\r\n\"\\]\\-\\\\]");
static std::unordered_map<char, std::string> GRAMMAR_LITERAL_ESCAPES = {
    {'\r', "\\r"}, {'\n', "\\n"}, {'"', "\\\""}, {'-', "\\-"}, {']', "\\]"}, {'\\', "\\\\"}
};

static const int MAX_PATTERN_DEPTH = 100;

static std::unordered_set<char> NON_LITERAL_SET = {'|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?', '^', '$'};
static std::unordered_set<char> ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = {'^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?'};

static std::string replacePattern(const std::string & input, const std::regex & regex, const std::function<std::string(const std::smatch  &)> & replacement) {
    std::smatch match;
    std::string result;

    std::string::const_iterator searchStart(input.cbegin());
    std::string::const_iterator searchEnd(input.cend());

    while (std::regex_search(searchStart, searchEnd, match, regex)) {
        result.append(searchStart, searchStart + match.position());
        result.append(replacement(match));
        searchStart = match.suffix().first;
    }

    result.append(searchStart, searchEnd);

    return result;
}

static std::string format_literal(const std::string & literal) {
    std::string escaped = replacePattern(literal, GRAMMAR_LITERAL_ESCAPE_RE, [&](const std::smatch & match) {
        char c = match.str()[0];
        return GRAMMAR_LITERAL_ESCAPES.at(c);
    });
    return "\"" + escaped + "\"";
}

std::string gbnf_format_literal(const std::string & literal) { return format_literal(literal); }

static size_t gbnf_escape_length(const std::string & pattern, size_t pos) {
    if (pos + 1 >= pattern.length() || pattern[pos] != '\\') {
        return 0;
    }
    size_t n_hex = 0;
    switch (pattern[pos + 1]) {
        case 'x': n_hex = 2; break;
        case 'u': n_hex = 4; break;
        case 'U': n_hex = 8; break;
        // keep in sync with parse_char() in src/llama-grammar.cpp
        case 't': case 'r': case 'n': case '\\': case '"': case '[': case ']': case '-':
            return 2;
        default:
            return 0;
    }
    if (pos + 2 + n_hex > pattern.length()) {
        return 0;
    }
    for (size_t i = pos + 2; i < pos + 2 + n_hex; i++) {
        char h = pattern[i];
        if (!((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'))) {
            return 0;
        }
    }
    return 2 + n_hex;
}

class common_chat_schema_converter {
private:
    friend std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options);
    bool _dotall;
    std::map<std::string, std::string> _rules;
    std::unordered_set<std::string> _refs_being_resolved;
    std::vector<std::string> _errors;
    std::vector<std::string> _warnings;

    template <typename T>
    static const T & as(const common_chat_schema & node) {
        return static_cast<const T &>(node);
    }

    std::string _add_rule(const std::string & name, const std::string & rule) {
        std::string esc_name = regex_replace(name, INVALID_RULE_CHARS_RE, "-");
        if (_rules.find(esc_name) == _rules.end() || _rules[esc_name] == rule) {
            _rules[esc_name] = rule;
            return esc_name;
        }
        int i = 0;
        while (_rules.find(esc_name + std::to_string(i)) != _rules.end() && _rules[esc_name + std::to_string(i)] != rule) {
            i++;
        }
        std::string key = esc_name + std::to_string(i);
        _rules[key] = rule;
        return key;
    }

    std::string _generate_union_rule(const std::string & name, const std::vector<common_chat_schema_ptr> & alt_schemas) {
        std::vector<std::string> rules;
        rules.reserve(alt_schemas.size());
        for (size_t i = 0; i < alt_schemas.size(); i++) {
            rules.push_back(visit(*alt_schemas[i], name + (name.empty() ? "alternative-" : "-") + std::to_string(i)));
        }
        return string_join(rules, " | ");
    }

    // thrown when the pattern is a valid regex with no grammar equivalent
    struct unsupported_pattern : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // thrown when the pattern is not a valid regex
    struct invalid_pattern : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    std::string _visit_pattern(const std::string & pattern, const std::string & name) {
        auto rules_snapshot = _rules;
        try {
            return _pattern_to_rule(pattern, name);
        } catch (const unsupported_pattern & err) {
            // revert rules
            _rules = std::move(rules_snapshot);
            _warnings.push_back("pattern " + pattern + " is not supported (" + err.what() + "), accepting any string");
            return _add_rule(name, _add_primitive("string", PRIMITIVE_RULES.at("string")));
        } catch (const invalid_pattern & err) {
            _rules = std::move(rules_snapshot);
            _errors.push_back("Invalid pattern " + pattern + ": " + err.what());
            return "";
        }
    }

    std::string _pattern_to_rule(const std::string & pattern, const std::string & name) {
        if (pattern.length() < 2 || pattern.front() != '^' || pattern.back() != '$') {
            throw unsupported_pattern("not anchored with '^' and '$'");
        }
        std::string sub_pattern = pattern.substr(1, pattern.length() - 2);
        std::unordered_map<std::string, std::string> sub_rule_ids;

        size_t i = 0;
        size_t length = sub_pattern.length();
        int paren_depth = 0;

        using literal_or_rule = std::pair<std::string, bool>;
        auto to_rule = [&](const literal_or_rule & ls) {
            auto is_literal = ls.second;
            auto s = ls.first;
            return is_literal ? "\"" + s + "\"" : s;
        };
        std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
            std::vector<literal_or_rule> seq;

            auto get_dot = [&]() {
                std::string rule;
                if (_dotall) {
                    rule = "[\\U00000000-\\U0010FFFF]";
                } else {
                    rule = "[^\\x0A\\x0D]";
                }
                return _add_rule("dot", rule);
            };

            // Joins the sequence, merging consecutive literals together.
            auto join_seq = [&]() {
                std::vector<literal_or_rule> ret;

                std::string literal;
                auto flush_literal = [&]() {
                    if (literal.empty()) {
                        return false;
                    }
                    ret.emplace_back(literal, true);
                    literal.clear();
                    return true;
                };

                for (const auto & item : seq) {
                    auto is_literal = item.second;
                    if (is_literal) {
                        literal += item.first;
                    } else {
                        flush_literal();
                        ret.push_back(item);
                    }
                }
                flush_literal();

                std::vector<std::string> results;
                results.reserve(ret.size());
                for (const auto & item : ret) {
                    results.push_back(to_rule(item));
                }
                return std::make_pair(string_join(results, " "), false);
            };

            while (i < length) {
                char c = sub_pattern[i];
                if (c == '.') {
                    seq.emplace_back(get_dot(), false);
                    i++;
                } else if (c == '(') {
                    i++;
                    if (i < length && sub_pattern[i] == '?') {
                        if (i + 1 < length && sub_pattern[i + 1] == ':') {
                            i += 2; // skip "?:" for non-capturing group, treat as regular group
                        } else {
                            // lookaround, named group, inline flags, ...
                            throw unsupported_pattern("unsupported group syntax");
                        }
                    }
                    paren_depth++;
                    if (paren_depth > MAX_PATTERN_DEPTH) {
                        throw unsupported_pattern("pattern nesting too deep");
                    }
                    seq.emplace_back("(" + to_rule(transform()) + ")", false);
                } else if (c == ')') {
                    i++;
                    if (paren_depth == 0) {
                        throw invalid_pattern("unbalanced parentheses");
                    }
                    paren_depth--;
                    return join_seq();
                } else if (c == '^' || c == '$') {
                    throw unsupported_pattern("anchor inside the pattern");
                } else if (c == '[') {
                    std::string square_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != ']') {
                        if (sub_pattern[i] == '\\') {
                            auto escape_length = gbnf_escape_length(sub_pattern, i);
                            if (escape_length == 0) {
                                throw unsupported_pattern("unsupported escape in character class: " + sub_pattern.substr(i, 2));
                            }
                            square_brackets += sub_pattern.substr(i, escape_length);
                            i += escape_length;
                        } else {
                            square_brackets += sub_pattern[i];
                            i++;
                        }
                    }
                    if (i >= length) {
                        throw invalid_pattern("unterminated character class");
                    }
                    square_brackets += ']';
                    i++;
                    seq.emplace_back(square_brackets, false);
                } else if (c == '|') {
                    seq.emplace_back("|", false);
                    i++;
                } else if (c == '*' || c == '+' || c == '?') {
                    if (seq.empty()) {
                        throw invalid_pattern("nothing to repeat");
                    }
                    seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
                    i++;
                } else if (c == '{') {
                    std::string curly_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != '}') {
                        curly_brackets += sub_pattern[i];
                        i++;
                    }
                    if (i >= length) {
                        throw unsupported_pattern("unterminated curly brackets");
                    }
                    curly_brackets += '}';
                    i++;
                    auto nums = string_split(curly_brackets.substr(1, curly_brackets.length() - 2), ",");
                    int min_times = 0;
                    int max_times = std::numeric_limits<int>::max();
                    if (nums.size() != 1 && nums.size() != 2) {
                        throw unsupported_pattern("wrong number of values in curly brackets");
                    }
                    try {
                        if (nums.size() == 1) {
                            min_times = max_times = std::stoi(nums[0]);
                        } else {
                            if (!nums[0].empty()) {
                                min_times = std::stoi(nums[0]);
                            }
                            if (!nums[1].empty()) {
                                max_times = std::stoi(nums[1]);
                            }
                        }
                    } catch (const std::logic_error &) {
                        throw unsupported_pattern("invalid number in curly brackets");
                    }
                    if (seq.empty()) {
                        throw invalid_pattern("nothing to repeat");
                    }
                    auto &last = seq.back();
                    auto &sub = last.first;
                    auto sub_is_literal = last.second;

                    if (!sub_is_literal) {
                        std::string & sub_id = sub_rule_ids[sub];
                        if (sub_id.empty()) {
                            sub_id = _add_rule(name + "-" + std::to_string(sub_rule_ids.size()), sub);
                        }
                        sub = sub_id;
                    }
                    seq.back().first = build_repetition(
                        sub_is_literal ? "\"" + sub + "\"" : sub,
                        min_times,
                        max_times,
                        ""
                    );
                    seq.back().second = false;
                } else {
                    std::string literal;
                    auto is_non_literal = [&](char c) {
                        return NON_LITERAL_SET.find(c) != NON_LITERAL_SET.end();
                    };
                    while (i < length) {
                        if (sub_pattern[i] == '\\') {
                            if (i == length - 1) {
                                throw invalid_pattern("trailing backslash");
                            }
                            char next = sub_pattern[i + 1];
                            if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.find(next) != ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.end()) {
                                i++;
                                literal += sub_pattern[i];
                                i++;
                            } else {
                                auto escape_length = gbnf_escape_length(sub_pattern, i);
                                if (escape_length == 0) {
                                    throw unsupported_pattern("unsupported escape: " + sub_pattern.substr(i, 2));
                                }
                                literal += sub_pattern.substr(i, escape_length);
                                i += escape_length;
                            }
                        } else if (sub_pattern[i] == '"') {
                            literal += "\\\"";
                            i++;
                        } else if (!is_non_literal(sub_pattern[i]) &&
                                (i == length - 1 || literal.empty() || sub_pattern[i + 1] == '.' || !is_non_literal(sub_pattern[i + 1]))) {
                            literal += sub_pattern[i];
                            i++;
                        } else {
                            break;
                        }
                    }
                    if (literal.empty()) { // nothing was consumed, ex. a stray ']' or '}'
                        throw unsupported_pattern(std::string("unsupported character: ") + c);
                    }
                    seq.emplace_back(literal, true);
                }
            }
            return join_seq();
        };

        auto rule = to_rule(transform());
        if (paren_depth != 0) {
            throw invalid_pattern("unbalanced parentheses");
        }

        return _add_rule(name, "\"\\\"\" (" + rule + ") \"\\\"\"");
    }

    /*
        Returns a rule that matches a JSON string that is none of the provided strings

        not_strings({"a"})
            -> ["] ( [a] char+ | [^"a] char* )? ["]
        not_strings({"and", "also"})
            -> ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["]
    */
    std::string _not_strings(const std::vector<std::string> & strings) {
        common_trie trie(strings);

        std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
        std::ostringstream out;
        out << "[\"] ( ";
        std::function<void(size_t)> visit = [&](size_t idx) {
            const auto & node = trie.nodes[idx];
            std::string rejects;
            auto first = true;
            for (const auto & [cpt, child] : node.children) {
                std::string c = common_unicode_cpt_to_utf8(cpt);
                rejects += c;
                if (first) {
                    first = false;
                } else {
                    out << " | ";
                }
                out << "[" << c << "]";
                if (!trie.nodes[child].children.empty()) {
                    out << " (";
                    visit(child);
                    out << ")";
                } else {
                    out << " " << char_rule << "+";
                }
            }
            if (!node.children.empty()) {
                out << " | [^\"" << rejects << "] " << char_rule << "*";
            }
        };
        visit(0);

        out << " )";
        if (trie.nodes[0].pattern < 0) {
            out << "?";
        }
        out << " [\"]";
        return out.str();
    }

    std::string _resolve_ref(const common_chat_schema_ref & schema) {
        auto it = schema.ref.find('#');
        std::string ref_fragment = it != std::string::npos ? schema.ref.substr(it + 1) : schema.ref;
        static const std::regex nonalphanumeric_regex(R"([^a-zA-Z0-9-]+)");
        std::string ref_name = "ref" + std::regex_replace(ref_fragment, nonalphanumeric_regex, "-");
        if (_rules.find(ref_name) == _rules.end() && _refs_being_resolved.find(schema.ref) == _refs_being_resolved.end()) {
            if (!schema.target) {
                _errors.push_back("Unresolved $ref " + schema.ref);
                return "";
            }
            _refs_being_resolved.insert(schema.ref);
            ref_name = visit(*schema.target, ref_name);
            _refs_being_resolved.erase(schema.ref);
        }
        return ref_name;
    }

    std::string _build_object_rule(
        const std::vector<std::pair<std::string, const common_chat_schema *>> & properties,
        const std::unordered_set<std::string> & required,
        const std::string & name,
        const common_chat_schema * additional_properties)
    {
        std::vector<std::string> required_props;
        std::vector<std::string> optional_props;
        std::unordered_map<std::string, std::string> prop_kv_rule_names;
        std::vector<std::string> prop_names;
        for (const auto & kv : properties) {
            const auto &prop_name = kv.first;
            const auto &prop_schema = kv.second;

            std::string prop_rule_name = visit(*prop_schema, name + (name.empty() ? "" : "-") + prop_name);
            prop_kv_rule_names[prop_name] = _add_rule(
                name + (name.empty() ? "" : "-") + prop_name + "-kv",
                format_literal(json(prop_name).dump()) + " space \":\" space " + prop_rule_name
            );
            if (required.find(prop_name) != required.end()) {
                required_props.push_back(prop_name);
            } else {
                optional_props.push_back(prop_name);
            }
            prop_names.push_back(prop_name);
        }
        if (additional_properties) {
            std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
            std::string value_rule =
                additional_properties->kind() != common_chat_schema::KIND_ANY ? visit(*additional_properties, sub_name + "-value")
                : _add_primitive("value", PRIMITIVE_RULES.at("value"));

            auto key_rule =
                prop_names.empty() ? _add_primitive("string", PRIMITIVE_RULES.at("string"))
                : _add_rule(sub_name + "-k", _not_strings(prop_names));
            std::string kv_rule = _add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
            prop_kv_rule_names["*"] = kv_rule;
            optional_props.push_back("*");
        }

        if (required_props.empty() && optional_props.empty()) {
            return "\"{\" space \"}\"";
        }

        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); i++) {
            if (i > 0) {
                rule += " \",\" space ";
            }
            rule += prop_kv_rule_names[required_props[i]];
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) {
                rule += " \",\" space ( ";
            }

            std::function<std::string(const std::vector<std::string> &, bool)> get_recursive_refs = [&](const std::vector<std::string> & ks, bool first_is_optional) {
                std::string res;
                if (ks.empty()) {
                    return res;
                }
                const std::string& k = ks[0];
                std::string kv_rule_name = prop_kv_rule_names[k];
                std::string comma_ref = "( \",\" space " + kv_rule_name + " )";
                if (first_is_optional) {
                    res = comma_ref + (k == "*" ? "*" : "?");
                } else {
                    res = kv_rule_name + (k == "*" ? " " + comma_ref + "*" : "");
                }
                if (ks.size() > 1) {
                    res += " " + _add_rule(
                        name + (name.empty() ? "" : "-") + k + "-rest",
                        get_recursive_refs(std::vector<std::string>(ks.begin() + 1, ks.end()), true)
                    );
                }
                return res;
            };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) {
                    rule += " | ";
                }
                rule += get_recursive_refs(std::vector<std::string>(optional_props.begin() + i, optional_props.end()), false);
            }
            if (!required_props.empty()) {
                rule += " )";
            }
            rule += " )?";
        }

        rule += " space \"}\"";

        return rule;
    }

    std::string _add_primitive(const std::string & name, const BuiltinRule & rule) {
        auto n = _add_rule(name, rule.content);
        for (const auto & dep : rule.deps) {
            BuiltinRule dep_rule;
            auto it = PRIMITIVE_RULES.find(dep);
            if (it == PRIMITIVE_RULES.end()) {
                it = STRING_FORMAT_RULES.find(dep);
                if (it == STRING_FORMAT_RULES.end()) {
                    _errors.push_back("Rule " + dep + " not known");
                    continue;
                }
            }
            if (_rules.find(dep) == _rules.end()) {
                _add_primitive(dep, it->second);
            }
        }
        return n;
    }

public:
    explicit common_chat_schema_converter(bool dotall) : _dotall(dotall) {
        _rules["space"] = SPACE_RULE;
    }

    std::string add_schema(const std::string & name, const common_chat_schema & schema) {
        return visit(schema, name);
    }

    static std::string _generate_constant_rule(const json & value) {
        return format_literal(value.dump());
    }

    std::string _visit_primitive(const std::string & rule_name, const std::string & type) {
        return _add_primitive(rule_name == "root" ? "root" : type, PRIMITIVE_RULES.at(type));
    }

    std::string _visit_all_of(const common_chat_schema_all_of & schema, const std::string & name, const std::string & rule_name) {
        std::unordered_set<std::string> required;
        std::vector<std::pair<std::string, const common_chat_schema *>> properties;
        std::map<std::string, size_t> enum_values;
        std::function<void(const common_chat_schema &, bool)> add_component = [&](const common_chat_schema & comp, bool is_required) {
            if (comp.kind() == common_chat_schema::KIND_REF) {
                if (const auto * target = as<common_chat_schema_ref>(comp).target) {
                    add_component(*target, is_required);
                }
            } else if (comp.kind() == common_chat_schema::KIND_OBJECT) {
                for (const auto & prop : as<common_chat_schema_object>(comp).properties) {
                    properties.emplace_back(prop.name, prop.schema.get());
                    if (is_required) {
                        required.insert(prop.name);
                    }
                }
            } else if (comp.kind() == common_chat_schema::KIND_ENUM) {
                for (const auto & v : as<common_chat_schema_enum>(comp).values) {
                    enum_values[_generate_constant_rule(v)] += 1;
                }
            }
        };
        for (const auto & child : schema.children) {
            if (child->kind() == common_chat_schema::KIND_ANY_OF) {
                for (const auto & alt : as<common_chat_schema_any_of>(*child).children) {
                    add_component(*alt, false);
                }
            } else {
                add_component(*child, true);
            }
        }
        if (!enum_values.empty()) {
            std::vector<std::string> enum_intersection;
            for (const auto & p : enum_values) {
                if (p.second == schema.children.size()) {
                    enum_intersection.push_back(p.first);
                }
            }
            if (!enum_intersection.empty()) {
                return _add_rule(rule_name, "(" + string_join(enum_intersection, " | ") + ")");
            }
        }
        return _add_rule(rule_name, _build_object_rule(properties, required, name, nullptr));
    }

    std::string visit(const common_chat_schema & schema, const std::string & name) {
        std::string rule_name = is_reserved_name(name) ? name + "-" : name.empty() ? "root" : name;
        std::string sub_name  = name + (name.empty() ? "" : "-");

        switch (schema.kind()) {
            case common_chat_schema::KIND_REF:
                return _add_rule(rule_name, _resolve_ref(as<common_chat_schema_ref>(schema)));
            case common_chat_schema::KIND_ANY_OF:
                return _add_rule(rule_name, _generate_union_rule(name, as<common_chat_schema_any_of>(schema).children));
            case common_chat_schema::KIND_ALL_OF:
                return _visit_all_of(as<common_chat_schema_all_of>(schema), name, rule_name);
            case common_chat_schema::KIND_CONST:
                return _add_rule(rule_name, _generate_constant_rule(as<common_chat_schema_const>(schema).value));
            case common_chat_schema::KIND_ENUM: {
                std::vector<std::string> enum_values;
                for (const auto & v : as<common_chat_schema_enum>(schema).values) {
                    enum_values.push_back(_generate_constant_rule(v));
                }
                return _add_rule(rule_name, "(" + string_join(enum_values, " | ") + ")");
            }
            case common_chat_schema::KIND_OBJECT: {
                const auto & obj = as<common_chat_schema_object>(schema);
                if (obj.properties.empty() && obj.additional_properties && obj.additional_properties->kind() == common_chat_schema::KIND_ANY) {
                    return _add_rule(rule_name, _add_primitive("object", PRIMITIVE_RULES.at("object")));
                }
                std::vector<std::pair<std::string, const common_chat_schema *>> properties;
                std::unordered_set<std::string> required;
                for (const auto & prop : obj.properties) {
                    properties.emplace_back(prop.name, prop.schema.get());
                    if (prop.required) {
                        required.insert(prop.name);
                    }
                }
                return _add_rule(rule_name, _build_object_rule(properties, required, name, obj.additional_properties.get()));
            }
            case common_chat_schema::KIND_TUPLE: {
                const auto & items = as<common_chat_schema_tuple>(schema).items;
                std::string rule = "\"[\" space ";
                for (size_t i = 0; i < items.size(); i++) {
                    if (i > 0) {
                        rule += " \",\" space ";
                    }
                    rule += visit(*items[i], sub_name + "tuple-" + std::to_string(i));
                }
                rule += " space \"]\"";
                return _add_rule(rule_name, rule);
            }
            case common_chat_schema::KIND_ARRAY: {
                const auto & arr = as<common_chat_schema_array>(schema);
                if (arr.items->kind() == common_chat_schema::KIND_ANY && arr.min_items == 0 && arr.max_items < 0) {
                    return _visit_primitive(rule_name, "array");
                }
                std::string item_rule_name = visit(*arr.items, sub_name + "item");
                int max_items = arr.max_items < 0 ? std::numeric_limits<int>::max() : arr.max_items;
                return _add_rule(rule_name, "\"[\" space " + build_repetition(item_rule_name, arr.min_items, max_items, "\",\" space") + " space \"]\"");
            }
            case common_chat_schema::KIND_STRING: {
                const auto & str = as<common_chat_schema_string>(schema);
                if (!str.pattern.empty()) {
                    return _visit_pattern(str.pattern, rule_name);
                }
                if (str.format == common_chat_schema::FORMAT_UUID) {
                    return _visit_primitive(rule_name, "uuid");
                }
                if (str.format != common_chat_schema::FORMAT_NONE) {
                    std::string prim_name = std::string(str.format == common_chat_schema::FORMAT_DATE ? "date" : str.format == common_chat_schema::FORMAT_TIME ? "time" : "date-time") + "-string";
                    return _add_rule(rule_name, _add_primitive(prim_name, STRING_FORMAT_RULES.at(prim_name)));
                }
                if (str.min_length > 0 || str.max_length >= 0) {
                    std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
                    int max_len = str.max_length < 0 ? std::numeric_limits<int>::max() : str.max_length;
                    return _add_rule(rule_name, "\"\\\"\" " + build_repetition(char_rule, str.min_length, max_len) + " \"\\\"\"");
                }
                return _visit_primitive(rule_name, "string");
            }
            case common_chat_schema::KIND_INTEGER: {
                const auto & i = as<common_chat_schema_integer>(schema);
                if (i.minimum == std::numeric_limits<int64_t>::min() && i.maximum == std::numeric_limits<int64_t>::max()) {
                    return _visit_primitive(rule_name, "integer");
                }
                std::stringstream out;
                out << "(";
                build_min_max_int(i.minimum, i.maximum, out);
                out << ")";
                return _add_rule(rule_name, out.str());
            }
            case common_chat_schema::KIND_NUMBER:
                return _visit_primitive(rule_name, "number");
            case common_chat_schema::KIND_BOOLEAN:
                return _visit_primitive(rule_name, "boolean");
            case common_chat_schema::KIND_NULL:
                return _visit_primitive(rule_name, "null");
            case common_chat_schema::KIND_ANY:
                return _add_rule(rule_name, _add_primitive("value", PRIMITIVE_RULES.at("value")));
        }
        return "";
    }

    void check_errors() {
        if (!_errors.empty()) {
            throw std::invalid_argument("JSON schema conversion failed:\n" + string_join(_errors, "\n"));
        }
        if (!_warnings.empty()) {
            fprintf(stderr, "WARNING: JSON schema conversion was incomplete: %s\n", string_join(_warnings, "; ").c_str());
        }
    }

    std::string format_grammar() {
        std::stringstream ss;
        for (const auto & kv : _rules) {
            ss << kv.first << " ::= " << kv.second << '\n';
        }
        return ss.str();
    }
};

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf) {
#ifdef LLAMA_USE_LLGUIDANCE
    if (!force_gbnf) {
        return "%llguidance {}\nstart: %json " + schema.dump();
    }
#else
    (void)force_gbnf;
#endif // LLAMA_USE_LLGUIDANCE
    try {
        return json_schema_to_grammar(common_chat_schema_from_json(schema));
    } catch (const std::runtime_error & e) {
        throw std::invalid_argument(std::string("JSON schema conversion failed:\n") + e.what());
    }
}

std::string json_schema_to_grammar(const common_chat_schema_document & schema) {
    common_chat_schema_converter converter(false);
    converter.visit(*schema.root, "");
    converter.check_errors();
    return converter.format_grammar();
}

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options) {
    common_chat_schema_converter converter(options.dotall);
    common_grammar_builder builder {
        /* .add_rule = */ [&](const std::string & name, const std::string & rule) {
            return converter._add_rule(name, rule);
        },
        /* .add_schema = */ [&](const std::string & name, const common_chat_schema & schema) {
            return converter.add_schema(name == "root" ? "" : name, schema);
        },
    };
    cb(builder);
    converter.check_errors();
    return converter.format_grammar();
}

// file: common/arg.cpp
#include "arg.h"

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "download.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "preset.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <filesystem>
#include <fstream>
#include <list>
#include <numeric>
#include <regex>
#include <set>
#include <string>
#include <system_error>
#include <thread> // for hardware_concurrency
#include <vector>

#ifndef __EMSCRIPTEN__
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#   if !defined(PATH_MAX)
#   define PATH_MAX MAX_PATH
#   endif
#elif defined(_AIX)
#include <sys/limits.h>
#else
#include <sys/syslimits.h>
#endif
#endif

#define LLAMA_MAX_URL_LENGTH 2084 // Maximum URL Length in Chrome: 2083

using json = common_json;
using namespace common_arg_utils;

static std::initializer_list<enum llama_example> mmproj_examples = {
    LLAMA_EXAMPLE_MTMD,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CLI,
    LLAMA_EXAMPLE_TTS,
};

static std::string read_file(const std::string & fname) {
    std::ifstream file(fname);
    if (!file) {
        throw std::runtime_error(string_format("error: failed to open file '%s'\n", fname.c_str()));
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

static const std::vector<common_arg> & get_common_arg_defs() {
    static const std::vector<common_arg> options = [] {
        common_params params;
        auto ctx = common_params_parser_init(params, LLAMA_EXAMPLE_SERVER, nullptr);
        return ctx.options;
    }();
    return options;
}

common_arg & common_arg::set_examples(std::initializer_list<enum llama_example> examples) {
    this->examples = examples;
    return *this;
}

common_arg & common_arg::set_excludes(std::initializer_list<enum llama_example> excludes) {
    this->excludes = excludes;
    return *this;
}

common_arg & common_arg::set_env(const char * env) {
    help = help + "\n(env: " + env + ")";
    this->env = env;
    return *this;
}

common_arg & common_arg::set_sampling() {
    is_sampling = true;
    return *this;
}

common_arg & common_arg::set_spec() {
    is_spec = true;
    return *this;
}

common_arg & common_arg::set_preset_only() {
    is_preset_only = true;
    return *this;
}

bool common_arg::in_example(enum llama_example ex) {
    return examples.find(ex) != examples.end();
}

bool common_arg::is_exclude(enum llama_example ex) {
    return excludes.find(ex) != excludes.end();
}

bool common_arg::get_value_from_env(std::string & output) const {
    if (env == nullptr) return false;
    if (!args_neg.empty()) {
        // for compatibility, we need to check LLAMA_ARG_NO_ env as well
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        char * neg_value = std::getenv(neg_env.c_str());
        if (neg_value) {
            output = "0"; // falsey
            return true;
        }
    }
    char * value = std::getenv(env);
    if (value) {
        output = value;
        return true;
    }
    return false;
}

bool common_arg::has_value_from_env() const {
    if (env != nullptr && !args_neg.empty()) {
        // for compatibility, we need to check LLAMA_ARG_NO_ env as well
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        if (std::getenv(neg_env.c_str())) {
            return true;
        }
    }
    return env != nullptr && std::getenv(env);
}

static std::vector<std::string> break_str_into_lines(std::string input, size_t max_char_per_line) {
    std::vector<std::string> result;
    std::istringstream iss(input);
    std::string line;
    auto add_line = [&](const std::string& l) {
        if (l.length() <= max_char_per_line) {
            result.push_back(l);
        } else {
            std::istringstream line_stream(l);
            std::string word, current_line;
            while (line_stream >> word) {
                if (current_line.length() + !current_line.empty() + word.length() > max_char_per_line) {
                    if (!current_line.empty()) result.push_back(current_line);
                    current_line = word;
                } else {
                    current_line += (!current_line.empty() ? " " : "") + word;
                }
            }
            if (!current_line.empty()) result.push_back(current_line);
        }
    };
    while (std::getline(iss, line)) {
        add_line(line);
    }
    return result;
}

std::string common_arg::to_string() const {
    // params for printing to console
    const static int n_leading_spaces = 40;
    const static int n_char_per_line_help = 70; // TODO: detect this based on current console
    std::string leading_spaces(n_leading_spaces, ' ');

    std::ostringstream ss;
    auto all_args = get_args(); // also contains args_neg
    for (const auto & arg : all_args) {
        if (arg == all_args.front()) {
            if (all_args.size() == 1) {
                ss << arg;
            } else {
                // first arg is usually abbreviation, we need padding to make it more beautiful
                auto tmp = std::string(arg) + ", ";
                auto spaces = std::string(std::max(0, 7 - (int)tmp.size()), ' ');
                ss << tmp << spaces;
            }
        } else {
            ss << arg << (arg != all_args.back() ? ", " : "");
        }
    }
    if (value_hint) ss << " " << value_hint;
    if (value_hint_2) ss << " " << value_hint_2;
    if (ss.tellp() > n_leading_spaces - 3) {
        // current line is too long, add new line
        ss << "\n" << leading_spaces;
    } else {
        // padding between arg and help, same line
        ss << std::string(leading_spaces.size() - ss.tellp(), ' ');
    }
    const auto help_lines = break_str_into_lines(help, n_char_per_line_help);
    for (const auto & line : help_lines) {
        ss << (&line == &help_lines.front() ? "" : leading_spaces) << line << "\n";
    }
    return ss.str();
}

std::vector<std::string> common_arg::get_args() const {
    std::vector<std::string> result;
    for (const auto & arg : args) {
        result.push_back(std::string(arg));
    }
    for (const auto & arg : args_neg) {
        result.push_back(std::string(arg));
    }
    return result;
}

std::vector<std::string> common_arg::get_env() const {
    std::vector<std::string> result;
    if (env) {
        result.push_back(std::string(env));
    }
    if (!args_neg.empty() && env) {
        // for compatibility, we need to add LLAMA_ARG_NO_ variant
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        result.push_back(neg_env);
    }
    return result;
}

//
// utils
//

// Helper function to parse tensor buffer override strings
static void parse_tensor_buffer_overrides(const std::string & value, std::vector<llama_model_tensor_buft_override> & overrides) {
    ggml_backend_load_all();

    std::map<std::string, ggml_backend_buffer_type_t> buft_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        if (buft) {
            buft_list[ggml_backend_buft_name(buft)] = buft;
        }
    }

    for (const auto & override : string_split<std::string>(value, ',')) {
        std::string::size_type pos = override.find('=');
        if (pos == std::string::npos) {
            throw std::invalid_argument("invalid value");
        }
        std::string tensor_name = override.substr(0, pos);
        std::string buffer_type = override.substr(pos + 1);

        if (buft_list.find(buffer_type) == buft_list.end()) {
            printf("Available buffer types:\n");
            for (const auto & it : buft_list) {
                printf("  %s\n", ggml_backend_buft_name(it.second));
            }
            throw std::invalid_argument("unknown buffer type");
        }
        // keep strings alive and avoid leaking memory by storing them in a static vector
        static std::list<std::string> buft_overrides;
        buft_overrides.push_back(tensor_name);
        overrides.push_back({buft_overrides.back().c_str(), buft_list.at(buffer_type)});
    }
}

static std::string clean_file_name(const std::string & fname) {
    std::string clean_fname = fname;
    string_replace_all(clean_fname, "\\", "_");
    string_replace_all(clean_fname, "/", "_");
    return clean_fname;
}

struct handle_model_result {
    bool found_mmproj = false;
    common_params_model mmproj;

    bool found_mtp = false;
    common_params_model mtp;

    bool found_preset = false;
    std::string preset_path;
};

const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};

static ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types) {
        if (ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

static std::string get_all_kv_cache_types() {
    std::ostringstream msg;
    for (const auto & type : kv_cache_types) {
        msg << ggml_type_name(type) << (&type == &kv_cache_types.back() ? "" : ", ");
    }
    return msg.str();
}

static bool parse_bool_value(const std::string & value) {
    if (is_truthy(value)) {
        return true;
    } else if (is_falsey(value)) {
        return false;
    } else {
        throw std::invalid_argument("invalid boolean value");
    }
}

[[noreturn]] static void arg_removed(const std::string & msg) {
    throw std::invalid_argument("the argument has been removed. " + msg);
}

//
// common_models_handler
//

static std::string get_default_local_path(const std::string & url) {
    auto f = string_split<std::string>(url, '#').front();
    f = string_split<std::string>(f, '?').front();
    return fs_get_cache_file(string_split<std::string>(f, '/').back());
}

static bool spec_types_is_default(const common_params & params) {
    return params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE};
}

common_models_handler common_models_handler_init(const common_params & params, llama_example curr_ex) {
    common_download_hf_plan plan;
    common_download_hf_plan plan_spec;
    common_download_opts opts;

    const bool spec_type_draft_mtp = std::find(params.speculative.types.begin(),
                                        params.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    const bool spec_type_draft_dflash = std::find(params.speculative.types.begin(),
                                           params.speculative.types.end(),
                                           COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != params.speculative.types.end();

    const bool spec_type_draft_eagle3 = std::find(params.speculative.types.begin(),
                                           params.speculative.types.end(),
                                           COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3) != params.speculative.types.end();

    const bool spec_type_draft_dspark = std::find(params.speculative.types.begin(),
                                           params.speculative.types.end(),
                                           COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) != params.speculative.types.end();

    // only download mmproj if the current example is using it
    bool use_mmproj = false;
    for (const auto & ex : mmproj_examples) {
        if (curr_ex == ex) {
            use_mmproj = true;
            break;
        }
    }

    opts.bearer_token    = params.hf_token;
    opts.offline         = params.offline;
    opts.download_mtp    = spec_type_draft_mtp;
    opts.download_eagle3 = spec_type_draft_eagle3;
    opts.download_dflash = spec_type_draft_dflash;
    opts.download_dspark = spec_type_draft_dspark;
    opts.download_mmproj = use_mmproj && !params.no_mmproj
                        && params.mmproj.path.empty() && params.mmproj.url.empty();

    if (!params.model.hf_repo.empty()) {
        plan = common_download_get_hf_plan(params.model, opts);
    }

    if (!params.speculative.draft.mparams.hf_repo.empty()) {
        // without a requested type, discover every sidecar the draft repo ships to infer the type later
        auto opts_spec = opts;
        if (spec_types_is_default(params)) {
            opts_spec.download_mtp    = true;
            opts_spec.download_dflash = true;
            opts_spec.download_eagle3 = true;
            opts_spec.download_dspark = true;
        }
        plan_spec = common_download_get_hf_plan(params.speculative.draft.mparams, opts_spec);
    }

    return common_models_handler{plan, plan_spec, opts};
}

bool common_models_handler_is_preset_repo(const common_models_handler & handler) {
    return !handler.plan.preset.url.empty();
}

static std::vector<common_download_task> build_url_tasks(const common_params_model & model, common_download_opts opts) {
    auto parts = common_download_get_all_parts(model.url);
    std::vector<common_download_task> tasks;

    // single-part: download straight to model.path if the user gave one (-m), else the cache default
    if (parts.size() == 1) {
        common_download_task task;
        task.url        = parts[0];
        task.local_path = model.path.empty() ? get_default_local_path(parts[0]) : model.path;
        task.opts       = opts;
        tasks.push_back(std::move(task));
        return tasks;
    }

    // multi-part: place each part under the user's -m directory (if given), else the cache default
    std::string base_dir;
    if (!model.path.empty()) {
        auto pos = model.path.rfind('/');
        base_dir = pos == std::string::npos ? std::string(".") : model.path.substr(0, pos);
    }

    for (const auto & part : parts) {
        common_download_task task;
        task.url  = part;
        task.opts = opts;

        std::string local = get_default_local_path(part);
        if (!base_dir.empty()) {
            auto pos = local.rfind('/');
            std::string name = pos == std::string::npos ? local : local.substr(pos + 1);
            local = base_dir + "/" + name;
        }
        task.local_path = local;
        tasks.push_back(std::move(task));
    }
    return tasks;
}

void common_models_handler_apply(common_models_handler & handler, common_params & params, common_download_callback * callback) {
    std::vector<common_download_task> tasks;

    auto & plan      = handler.plan;
    auto & plan_spec = handler.plan_spec;

    auto opts = handler.opts; // copy
    opts.callback = callback;

    // handle plain "url" if needed
    auto handle_url = [&](common_params_model & model) {
        if (!model.url.empty()) {
            if (model.path.empty()) {
                model.path = get_default_local_path(model.url);
            }
        }
    };
    handle_url(params.model);
    handle_url(params.mmproj);
    handle_url(params.speculative.draft.mparams);

    // optionally, if docker repo is set, resolve it
    if (!params.model.docker_repo.empty()) {
        params.model.url  = common_docker_resolve_model(params.model.docker_repo);
        params.model.path = get_default_local_path(params.model.url);
    }

    // handle plain "url" tasks (non-hf)
    if (!params.model.url.empty()) {
        auto url_tasks = build_url_tasks(params.model, opts);
        // the first part is what gets loaded, so point params.model.path at it
        if (!url_tasks.empty()) {
            std::string first_path = url_tasks.front().local_path;
            url_tasks.front().on_done = [&, first_path]() { params.model.path = first_path; };
        }
        for (auto & task : url_tasks) {
            tasks.push_back(std::move(task));
        }
    }
    if (!params.mmproj.url.empty()) {
        common_download_task task;
        task.url        = params.mmproj.url;
        task.local_path = params.mmproj.path;
        task.opts       = opts;
        tasks.push_back(task);
    }
    bool had_spec_url = false;
    if (!params.speculative.draft.mparams.url.empty()) {
        common_download_task task;
        task.url        = params.speculative.draft.mparams.url;
        task.local_path = params.speculative.draft.mparams.path;
        task.opts       = opts;
        tasks.push_back(task);
        had_spec_url = true;
    }

    // handle hf_plan tasks
    auto add_tasks = [&opts, &tasks](const hf_cache::hf_files  & model_files,
                                    const hf_cache::hf_file    & primary,
                                    common_params_model        & model) {
        for (size_t i = 0; i < model_files.size(); ++i) {
            auto & model_file = model_files[i];
            bool is_primary = (model_file.path == primary.path);
            tasks.emplace_back(model_file, opts, [&, is_primary]() {
                if (is_primary) {
                    // the primary file is the first split (00001-of), use it as model path
                    model.path = hf_cache::finalize_file(model_file);
                } else {
                    hf_cache::finalize_file(model_file);
                }
            });
        }
    };

    // an explicit draft file selection (e.g. -md with -hfd) disables the sidecar resolution of the draft repo
    if (!params.speculative.draft.mparams.hf_file.empty()) {
        plan_spec.mtp    = {};
        plan_spec.dflash = {};
        plan_spec.eagle3 = {};
        plan_spec.dspark = {};
    }

    // infer the speculative type from the sidecar shipped by the draft repo when none is requested
    if (spec_types_is_default(params)) {
        if (!plan_spec.mtp.local_path.empty()) {
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            plan_spec.dspark = {};
            plan_spec.dflash = {};
            plan_spec.eagle3 = {};
        } else if (!plan_spec.dspark.local_path.empty()) {
            // dspark outranks dflash, its sidecar carries the extra Markov head
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
            plan_spec.dflash = {};
            plan_spec.eagle3 = {};
        } else if (!plan_spec.dflash.local_path.empty()) {
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
            plan_spec.eagle3 = {};
        } else if (!plan_spec.eagle3.local_path.empty()) {
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 };
        }
    }

    // infer the speculative type from the draft GGUF metadata when none is requested
    // note: reads only the first split - sharded drafts need an explicit --spec-type
    if (spec_types_is_default(params) && !params.speculative.draft.mparams.path.empty()) {
        const auto types_gguf = common_speculative_types_from_gguf(params.speculative.draft.mparams.path);
        if (!types_gguf.empty()) {
            params.speculative.types = types_gguf;
        }
    }

    // when a sidecar type is requested, the draft repo resolves to its sidecar instead of a full model
    const bool spec_sidecar_found = !plan_spec.mtp.local_path.empty() ||
                                    !plan_spec.dflash.local_path.empty() ||
                                    !plan_spec.eagle3.local_path.empty() ||
                                    !plan_spec.dspark.local_path.empty();
    if (!plan_spec.mtp.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan_spec.mtp, opts, [&]() {
            // only use the discovered MTP head when no draft path is set yet
            if (params.speculative.draft.mparams.path.empty()) {
                params.speculative.draft.mparams.path = hf_cache::finalize_file(plan_spec.mtp);
            } else {
                hf_cache::finalize_file(plan_spec.mtp);
            }
        });
    }
    if (!plan_spec.dflash.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan_spec.dflash, opts, [&]() {
            // only use the discovered DFlash sidecar when no draft path is set yet
            if (params.speculative.draft.mparams.path.empty()) {
                params.speculative.draft.mparams.path = hf_cache::finalize_file(plan_spec.dflash);
            } else {
                hf_cache::finalize_file(plan_spec.dflash);
            }
        });
    }
    if (!plan_spec.eagle3.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan_spec.eagle3, opts, [&]() {
            // only use the discovered Eagle3 sidecar when no draft path is set yet
            if (params.speculative.draft.mparams.path.empty()) {
                params.speculative.draft.mparams.path = hf_cache::finalize_file(plan_spec.eagle3);
            } else {
                hf_cache::finalize_file(plan_spec.eagle3);
            }
        });
    }
    if (!plan_spec.dspark.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan_spec.dspark, opts, [&]() {
            // only use the discovered DSpark sidecar when no draft path is set yet
            if (params.speculative.draft.mparams.path.empty()) {
                params.speculative.draft.mparams.path = hf_cache::finalize_file(plan_spec.dspark);
            } else {
                hf_cache::finalize_file(plan_spec.dspark);
            }
        });
    }

    // a wired draft sidecar counts as an explicit draft for the main plan fallback below
    if (spec_sidecar_found) {
        had_spec_url = true;
    }

    // handle plan_spec (e.g. --spec-draft-hf)
    if (!plan_spec.model_files.empty() && !had_spec_url && !spec_sidecar_found) {
        add_tasks(plan_spec.model_files, plan_spec.primary, params.speculative.draft.mparams);
        had_spec_url = true;
    }

    if (!plan.model_files.empty()) {
        add_tasks(plan.model_files, plan.primary, params.model);
    }
    if (!plan.mmproj.local_path.empty()) {
        tasks.emplace_back(plan.mmproj, opts, [&]() {
            params.mmproj.path = hf_cache::finalize_file(plan.mmproj);
        });
    }
    if (!plan.mtp.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan.mtp, opts, [&]() {
            // only fall back to the discovered MTP head when no draft was explicitly provided
            if (params.speculative.draft.mparams.empty()) {
                params.speculative.draft.mparams.path = hf_cache::finalize_file(plan.mtp);
            } else {
                hf_cache::finalize_file(plan.mtp);
            }
        });
    }
    if (!plan.dflash.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan.dflash, opts, [&]() {
            // only fall back to the discovered DFlash sidecar when no draft was explicitly provided
            if (params.speculative.draft.mparams.empty()) {
                params.speculative.draft.mparams.path = hf_cache::finalize_file(plan.dflash);
            } else {
                hf_cache::finalize_file(plan.dflash);
            }
        });
    }
    if (!plan.eagle3.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan.eagle3, opts, [&]() {
            // only fall back to the discovered Eagle3 sidecar when no draft was explicitly provided
            if (params.speculative.draft.mparams.empty()) {
                params.speculative.draft.mparams.path = hf_cache::finalize_file(plan.eagle3);
            } else {
                hf_cache::finalize_file(plan.eagle3);
            }
        });
    }
    if (!plan.dspark.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan.dspark, opts, [&]() {
            // only fall back to the discovered DSpark sidecar when no draft was explicitly provided
            if (params.speculative.draft.mparams.empty()) {
                params.speculative.draft.mparams.path = hf_cache::finalize_file(plan.dspark);
            } else {
                hf_cache::finalize_file(plan.dspark);
            }
        });
    }
    if (!plan.preset.local_path.empty()) {
        tasks.emplace_back(plan.preset, opts, [&]() {
            // if HF repo is a preset repo, we simply run server in router mode with the preset.ini file
            params.models_preset_hf = params.model.hf_repo; // only for showing a warning
            params.models_preset    = hf_cache::finalize_file(plan.preset);
            params.model = common_params_model{}; // make sure to clear model, so server starts in router mode
        });
    }

    // run all tasks in parallel
    if (!params.offline) {
        // if duplicated files are found, only download once (but still call on_done for each task)
        std::unordered_map<std::string, common_download_task *> unique_tasks;
        for (auto & task : tasks) {
            auto it = unique_tasks.find(task.local_path);
            if (it == unique_tasks.end()) {
                unique_tasks[task.local_path] = &task;
            }
        }
        std::vector<common_download_task> unique_tasks_vec;
        for (auto & pair : unique_tasks) {
            LOG_DBG("download task: %s -> %s\n", pair.second->url.c_str(), pair.second->local_path.c_str());
            unique_tasks_vec.push_back(*pair.second);
        }
        common_download_run_tasks(unique_tasks_vec);
    }

    // download successful, update params with the downloaded paths
    for (const auto & task : tasks) {
        if (task.on_done) {
            task.on_done();
        }
    }
}

//
// CLI argument parsing functions
//

// apply config files (if present), a later file overrides an earlier one:
// 1. system-wide: /etc/llama.cpp/config.ini (%PROGRAMDATA%\llama.cpp\config.ini on windows)
// 2. user-level: ${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini (%APPDATA%\llama.cpp\config.ini on windows)
static void common_params_apply_system_config(common_params & params, llama_example ex) {
    std::vector<std::string> paths;

#if defined(_WIN32)
    const std::string program_data = common_get_env("PROGRAMDATA");
    if (!program_data.empty()) {
        paths.push_back(program_data + "\\llama.cpp\\config.ini");
    }
#else
    paths.push_back("/etc/llama.cpp/config.ini");
#endif

    try {
        paths.push_back(fs_get_config_directory() + "config.ini");
    } catch (const std::exception & e) {
        LOG_DBG("cannot read user-level config file, skipping: %s\n", e.what());
    }

    std::vector<std::string> found;
    for (const auto & path : paths) {
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            found.push_back(path);
        }
    }
    if (found.empty()) {
        return;
    }

    common_preset_context ctx(ex);
    ctx.ignore_unknown_keys = true; // the same config file is shared by all programs
    for (const auto & path : found) {
        LOG_INF("using config file: %s\n", path.c_str());
        common_preset global;
        common_presets presets = ctx.load_from_ini(path, global);
        global.apply_to_params(params);
        auto it = presets.find(COMMON_PRESET_DEFAULT_NAME);
        if (it != presets.end()) {
            it->second.apply_to_params(params);
        }
    }
}

static bool common_params_parse_ex(int argc, char ** argv, common_params_context & ctx_arg) {
    common_params & params = ctx_arg.params;

    // setup log directly from params.verbosity: see tools/cli/cli.cpp
    common_log_set_verbosity_thold(params.verbosity);

    // config file applies first, so env variables and CLI arguments override it
    common_params_apply_system_config(params, ctx_arg.ex);

    std::unordered_map<std::string, std::pair<common_arg *, bool>> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = {&opt, /* is_positive */ true};
        }
        for (const auto & arg : opt.args_neg) {
            arg_to_options[arg] = {&opt, /* is_positive */ false};
        }
    }

    // handle environment variables
    for (auto & opt : ctx_arg.options) {
        std::string value;
        if (opt.get_value_from_env(value)) {
            try {
                if (opt.handler_void && is_truthy(value)) {
                    opt.handler_void(params);
                }
                if (opt.handler_int) {
                    opt.handler_int(params, std::stoi(value));
                }
                if (opt.handler_bool) {
                    opt.handler_bool(params, parse_bool_value(value));
                }
                if (opt.handler_string) {
                    opt.handler_string(params, value);
                    continue;
                }
            } catch (std::exception & e) {
                throw std::invalid_argument(string_format(
                    "error while handling environment variable \"%s\": %s\n\n", opt.env, e.what()));
            }
        }
    }

    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };

    auto parse_cli_args = [&]() {
        std::set<std::string> seen_args;

        for (int i = 1; i < argc; i++) {
            const std::string arg_prefix = "--";

            std::string arg = argv[i];
            if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
                std::replace(arg.begin(), arg.end(), '_', '-');
            }
            if (arg_to_options.find(arg) == arg_to_options.end()) {
                throw std::invalid_argument(string_format("error: invalid argument: %s", arg.c_str()));
            }
            if (!seen_args.insert(arg).second) {
                const bool skip = (arg == "--spec-type");

                if (!skip) {
                    LOG_WRN("DEPRECATED: argument '%s' specified multiple times, use comma-separated values instead (only last value will be used)\n", arg.c_str());
                }
            }
            auto & tmp = arg_to_options[arg];
            auto opt = *tmp.first;
            bool is_positive = tmp.second;
            if (opt.has_value_from_env()) {
                fprintf(stderr, "warn: %s environment variable is set, but will be overwritten by command line argument %s\n", opt.env, arg.c_str());
            }
            try {
                if (opt.handler_void) {
                    opt.handler_void(params);
                    continue;
                }
                if (opt.handler_bool) {
                    opt.handler_bool(params, is_positive);
                    continue;
                }

                // arg with single value
                check_arg(i);
                std::string val = argv[++i];
                if (opt.handler_int) {
                    opt.handler_int(params, std::stoi(val));
                    continue;
                }
                if (opt.handler_string) {
                    opt.handler_string(params, val);
                    continue;
                }

                // arg with 2 values
                check_arg(i);
                std::string val2 = argv[++i];
                if (opt.handler_str_str) {
                    opt.handler_str_str(params, val, val2);
                    continue;
                }
            } catch (std::exception & e) {
                throw std::invalid_argument(string_format(
                    "error while handling argument \"%s\": %s\n\n"
                    "usage:\n%s\n\nto show complete usage, run with -h",
                    arg.c_str(), e.what(), opt.to_string().c_str()));
            }
        }
    };

    // parse all CLI args now, so that -hf is available below for remote preset resolution
    parse_cli_args();

    postprocess_cpu_params(params.cpuparams,       nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    postprocess_cpu_params(params.speculative.draft.cpuparams,       &params.cpuparams);
    postprocess_cpu_params(params.speculative.draft.cpuparams_batch, &params.cpuparams_batch);

    // default the mmproj device to the global device selection if not set explicitly with -mmdev
    if (params.mmproj_use_gpu && params.mmproj_device == nullptr && !params.devices.empty()) {
        params.mmproj_device = params.devices.front();
        params.mmproj_use_gpu = params.mmproj_device != nullptr;
    }

    if (params.prompt_cache_all && (params.interactive || params.interactive_first)) {
        throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
    }

    const bool skip_model_download =
        // server will call common_params_handle_models() later, so we skip it here
        ctx_arg.ex == LLAMA_EXAMPLE_SERVER ||
        // download calls common_params_handle_models() itself and prints the paths
        ctx_arg.ex == LLAMA_EXAMPLE_DOWNLOAD ||
        // export_graph_ops loads only metadata
        ctx_arg.ex == LLAMA_EXAMPLE_EXPORT_GRAPH_OPS;

    if (!skip_model_download) {
        // handle model and download
        common_models_handler handler = common_models_handler_init(params, ctx_arg.ex);
        common_models_handler_apply(handler, params);

        // model is required (except for server)
        // TODO @ngxson : maybe show a list of available models in CLI in this case
        bool can_skip_model = params.usage || params.completion || !params.server_base.empty();
        if (!can_skip_model && params.model.path.empty()) {
            throw std::invalid_argument("error: --model is required\n");
        }
    }

    if (params.escape) {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        for (auto & antiprompt : params.antiprompt) {
            string_process_escapes(antiprompt);
        }
        for (auto & seq_breaker : params.sampling.dry_sequence_breakers) {
            string_process_escapes(seq_breaker);
        }
    }

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    const bool mcp_enabled = !params.mcp_servers_config.empty() || !params.mcp_servers_json.empty();
    if ((!params.server_tools.empty() || mcp_enabled) && !params.cors_origins_explicit) {
        LOG_WRN("server tools or MCP servers are enabled, using localhost as default CORS origin (change via --cors-origins)\n");
        params.cors_origins = "localhost";
    }

    // pad tensor_buft_overrides for llama_params_fit:
    const size_t ntbo = llama_max_tensor_buft_overrides();
    while (params.tensor_buft_overrides.size() < ntbo) {
        params.tensor_buft_overrides.push_back({nullptr, nullptr});
    }

    if (!params.speculative.draft.tensor_buft_overrides.empty()) {
        params.speculative.draft.tensor_buft_overrides.push_back({nullptr, nullptr});
    }

    if (!params.chat_template.empty() && !common_chat_verify_template(params.chat_template, params.use_jinja)) {
        throw std::runtime_error(string_format(
            "error: the supplied chat template is not supported: %s%s\n",
            params.chat_template.c_str(),
            params.use_jinja ? "" : "\nnote: llama.cpp was started without --jinja, we only support commonly used templates"
        ));
    }

    // if the preserve_reasoning kwarg was not specified explicitly, enable it by default
    if (!params.default_template_kwargs.count("preserve_reasoning")) {
        params.default_template_kwargs["preserve_reasoning"] = "true";
    }

    return true;
}

static void common_params_print_usage(common_params_context & ctx_arg) {
    auto print_options = [](std::vector<common_arg *> & options) {
        for (common_arg * opt : options) {
            printf("%s", opt->to_string().c_str());
        }
    };

    std::vector<common_arg *> common_options;
    std::vector<common_arg *> sampling_options;
    std::vector<common_arg *> spec_options;
    std::vector<common_arg *> specific_options;
    for (auto & opt : ctx_arg.options) {
        // in case multiple LLAMA_EXAMPLE_* are set, we prioritize the LLAMA_EXAMPLE_* matching current example
        if (opt.is_sampling) {
            sampling_options.push_back(&opt);
        } else if (opt.is_spec) {
            spec_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }
    bool first = true;
    auto print_section = [&](const char * header, std::vector<common_arg *> & options) {
        if (options.empty()) {
            return;
        }
        printf("%s----- %s -----\n\n", first ? "" : "\n\n", header);
        first = false;
        print_options(options);
    };
    print_section("common params",           common_options);
    print_section("sampling params",         sampling_options);
    print_section("speculative params",      spec_options);
    print_section("example-specific params", specific_options);
}

static void common_params_print_completion(common_params_context & ctx_arg) {
    std::vector<common_arg *> common_options;
    std::vector<common_arg *> sampling_options;
    std::vector<common_arg *> spec_options;
    std::vector<common_arg *> specific_options;

    for (auto & opt : ctx_arg.options) {
        if (opt.is_sampling) {
            sampling_options.push_back(&opt);
        } else if (opt.is_spec) {
            spec_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }

    printf("_llama_completions() {\n");
    printf("    local cur prev opts\n");
    printf("    COMPREPLY=()\n");
    printf("    cur=\"${COMP_WORDS[COMP_CWORD]}\"\n");
    printf("    prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n\n");

    printf("    opts=\"");
    auto print_options = [](const std::vector<common_arg *> & options) {
        for (const common_arg * opt : options) {
            for (const char * arg : opt->args) {
                printf("%s ", arg);
            }
        }
    };

    print_options(common_options);
    print_options(sampling_options);
    print_options(spec_options);
    print_options(specific_options);
    printf("\"\n\n");

    printf("    case \"$prev\" in\n");
    printf("        --model|-m)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.gguf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        --grammar-file)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.gbnf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        --chat-template-file)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.jinja' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        *)\n");
    printf("            COMPREPLY=( $(compgen -W \"${opts}\" -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("    esac\n");
    printf("}\n\n");

    std::set<std::string> executables = {
        "llama-batched",
        "llama-batched-bench",
        "llama-bench",
        "llama-cli",
        "llama-completion",
        "llama-convert-llama2c-to-ggml",
        "llama-cvector-generator",
        "llama-debug",
        "llama-diffusion-cli",
        "llama-embedding",
        "llama-eval-callback",
        "llama-export-lora",
        "llama-finetune",
        "llama-fit-params",
        "llama-gemma3-cli",
        "llama-gen-docs",
        "llama-gguf",
        "llama-gguf-hash",
        "llama-gguf-split",
        "llama-idle",
        "llama-imatrix",
        "llama-llava-cli",
        "llama-lookahead",
        "llama-lookup",
        "llama-lookup-create",
        "llama-lookup-merge",
        "llama-lookup-stats",
        "llama-minicpmv-cli",
        "llama-mtmd-cli",
        "llama-parallel",
        "llama-passkey",
        "llama-perplexity",
        "llama-q8dot",
        "llama-quantize",
        "llama-qwen2vl-cli",
        "llama-retrieval",
        "llama-save-load-state",
        "llama-server",
        "llama-simple",
        "llama-simple-chat",
        "llama-speculative",
        "llama-speculative-simple",
        "llama-tokenize",
        "llama-tts",
        "llama-vdot"
    };

    for (const auto& exe : executables) {
        printf("complete -F _llama_completions %s\n", exe.c_str());
    }
}

static std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    auto dev_names = string_split<std::string>(value, ',');
    if (dev_names.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);
    } else {
        ggml_backend_load_all();
        for (const auto & device : dev_names) {
            auto * dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                throw std::invalid_argument(string_format("invalid device: %s", device.c_str()));
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}

void common_print_available_devices() {
    constexpr size_t MiB = 1024 * 1024;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            devices.push_back(dev);
        }
    }
    printf("Available devices:\n");

    if (devices.empty()) {
        printf("  (none)\n");
        return;
    }
    for (auto * dev : devices) {
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / MiB, free / MiB);
    }
}

static void add_rpc_devices(const std::string & servers) {
    auto rpc_servers = string_split<std::string>(servers, ',');
    if (rpc_servers.empty()) {
        throw std::invalid_argument("no RPC servers specified");
    }
    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        throw std::invalid_argument("failed to find RPC backend");
    }
    typedef ggml_backend_reg_t (*ggml_backend_rpc_add_server_t)(const char * endpoint);
    ggml_backend_rpc_add_server_t ggml_backend_rpc_add_server_fn = (ggml_backend_rpc_add_server_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!ggml_backend_rpc_add_server_fn) {
        throw std::invalid_argument("failed to find RPC add server function");
    }
    for (const auto & server : rpc_servers) {
        auto reg = ggml_backend_rpc_add_server_fn(server.c_str());
        ggml_backend_register(reg);
    }
}

bool common_params_to_map(int argc, char ** argv, llama_example ex, std::map<common_arg, std::string> & out_map) {
    common_params dummy_params;
    common_params_context ctx_arg = common_params_parser_init(dummy_params, ex, nullptr);

    std::unordered_map<std::string, common_arg *> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = &opt;
        }
        for (const auto & arg : opt.args_neg) {
            arg_to_options[arg] = &opt;
        }
    }

    // TODO @ngxson : find a way to deduplicate this code

    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };

    std::set<std::string> seen_args;

    for (int i = 1; i < argc; i++) {
        const std::string arg_prefix = "--";

        std::string arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }
        if (arg_to_options.find(arg) == arg_to_options.end()) {
            throw std::invalid_argument(string_format("error: invalid argument: %s", arg.c_str()));
        }
        if (!seen_args.insert(arg).second) {
            const bool skip = (arg == "--spec-type");

            if (!skip) {
                LOG_WRN("DEPRECATED: argument '%s' specified multiple times, use comma-separated values instead (only last value will be used)\n", arg.c_str());
            }
        }
        auto opt = *arg_to_options[arg];
        std::string val;
        if (opt.value_hint == nullptr && opt.value_hint_2 == nullptr) {
            // bool arg (need to reverse the meaning for negative args)
            bool is_neg = std::find(opt.args_neg.begin(), opt.args_neg.end(), arg) != opt.args_neg.end();
            val = is_neg ? "0" : "1";
        }
        if (opt.value_hint != nullptr) {
            // arg with single value
            check_arg(i);
            val = argv[++i];
        }
        if (opt.value_hint_2 != nullptr) {
            // TODO: support arg with 2 values
            throw std::invalid_argument("error: argument with 2 values is not yet supported\n");
        }
        out_map[opt] = val;
    }

    return true;
}

#ifdef _WIN32
struct utf8_argv {
    std::vector<std::string> buf;
    std::vector<char*> ptrs;
};

static utf8_argv make_utf8_argv() {
    utf8_argv out;
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv) return out;

    out.buf.reserve(wargc);
    for (int i = 0; i < wargc; ++i) {
        int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        if (n <= 0) { out.buf.emplace_back(); continue; }
        auto& s = out.buf.emplace_back();
        s.resize(static_cast<size_t>(n - 1));
        (void)WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), n, nullptr, nullptr);
    }
    LocalFree(wargv);

    out.ptrs.reserve(out.buf.size() + 1);
    for (auto& s : out.buf) out.ptrs.push_back(s.data());
    out.ptrs.push_back(nullptr);
    return out;
}
#endif

bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
#ifdef _WIN32
    auto utf8 = make_utf8_argv();
    // repair argv only when it matches the process command line
    if (static_cast<int>(utf8.buf.size()) == argc) {
        argv = utf8.ptrs.data();
    }
#endif

    auto ctx_arg = common_params_parser_init(params, ex, print_usage);
    const common_params params_org = ctx_arg.params; // the example can modify the default params

    try {
        if (!common_params_parse_ex(argc, argv, ctx_arg)) {
            ctx_arg.params = params_org;
            return false;
        }
        if (ctx_arg.params.usage) {
            common_params_print_usage(ctx_arg);
            if (ctx_arg.print_usage) {
                ctx_arg.print_usage(argc, argv);
            }
            common_log_flush(common_log_main());
            exit(0);
        }
        if (ctx_arg.params.completion) {
            common_params_print_completion(ctx_arg);
            exit(0);
        }
        params.lr.init();
    } catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        ctx_arg.params = params_org;
        return false;
    } catch (std::exception & ex) {
        fprintf(stderr, "%s\n", ex.what());
        exit(1); // for other exceptions, we exit with status code 1
    }

    return true;
}

static std::string list_builtin_chat_templates() {
    std::vector<const char *> supported_tmpl;
    int32_t res = llama_chat_builtin_templates(nullptr, 0);
    supported_tmpl.resize(res);
    res = llama_chat_builtin_templates(supported_tmpl.data(), supported_tmpl.size());
    std::ostringstream msg;
    for (auto & tmpl : supported_tmpl) {
        msg << tmpl << (&tmpl == &supported_tmpl.back() ? "" : ", ");
    }
    return msg.str();
}

bool common_arg_utils::is_truthy(const std::string & value) {
    return value == "on" || value == "enabled" || value == "true" || value == "1";
}

bool common_arg_utils::is_falsey(const std::string & value) {
    return value == "off" || value == "disabled" || value == "false" || value == "0";
}

bool common_arg_utils::is_autoy(const std::string & value) {
    return value == "auto" || value == "-1";
}

// Simple CSV parser that handles quoted fields and escaped quotes
// example:
//    input:  value1,"value, with, commas","value with ""escaped"" quotes",value4
//    output: [value1] [value, with, commas] [value with "escaped" quotes] [value4]
static std::vector<std::string> parse_csv_row(const std::string& input) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (size_t i = 0; i < input.length(); ++i) {
        char ch = input[i];

        if (ch == '"') {
            if (!in_quotes) {
                // start of quoted field (only valid if at beginning of field)
                if (!field.empty()) {
                    // quote appeared in middle of unquoted field, treat as literal
                    field += '"';
                } else {
                    in_quotes = true; // start
                }
            } else {
                if (i + 1 < input.length() && input[i + 1] == '"') {
                    // escaped quote: ""
                    field += '"';
                    ++i; // skip the next quote
                } else {
                    in_quotes = false; // end
                }
            }
        } else if (ch == ',') {
            if (in_quotes) {
                field += ',';
            } else {
                fields.push_back(std::move(field));
                field.clear();
            }
        } else {
            field += ch;
        }
    }

    // Add the last field
    fields.push_back(std::move(field));

    return fields;
}

common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    // per-example default params
    // we define here to make sure it's included in llama-gen-docs
    if (ex == LLAMA_EXAMPLE_COMPLETION) {
        params.use_jinja = false;   // disable jinja by default
    } else if (ex == LLAMA_EXAMPLE_MTMD) {
        params.use_jinja = false;   // disable jinja by default
        params.sampling.temp = 0.2; // lower temp by default for better quality
    } else if (ex == LLAMA_EXAMPLE_SERVER) {
        params.n_parallel = -1;     // auto by default
    } else if (ex == LLAMA_EXAMPLE_TOKENIZE) {
        params.parse_special = true; // parse special tokens by default, like the old tokenize tool
    } else if (ex == LLAMA_EXAMPLE_TTS) {
        params.out_file = "output.wav";
        params.sampling.penalty_repeat = 1.05f;
        params.sampling.penalty_last_n = -1;
    }

    params.use_color = tty_can_use_colors();

    common_params_context ctx_arg(params);
    ctx_arg.print_usage = print_usage;
    ctx_arg.ex          = ex;

    std::string sampler_type_chars;
    std::string sampler_type_names;
    for (const auto & sampler : params.sampling.samplers) {
        sampler_type_chars += common_sampler_type_to_chr(sampler);
        sampler_type_names += common_sampler_type_to_str(sampler) + ";";
    }
    if (!sampler_type_names.empty()) {
        sampler_type_names.pop_back(); // remove last semicolon
    }

    /**
     * filter options by example
     * rules:
     * - all examples inherit options from LLAMA_EXAMPLE_COMMON
     * - if LLAMA_EXAMPLE_* is set (other than COMMON), we only show the option in the corresponding example
     * - if both {LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_*,} are set, we will prioritize the LLAMA_EXAMPLE_* matching current example
     */
    auto add_opt = [&](common_arg arg) {
        // download only exposes the handful of args explicitly tagged for it
        const bool inherit_common = ex != LLAMA_EXAMPLE_DOWNLOAD;
        if ((arg.in_example(ex) || (inherit_common && arg.in_example(LLAMA_EXAMPLE_COMMON))) && !arg.is_exclude(ex)) {
            ctx_arg.options.push_back(std::move(arg));
        }
    };

    add_opt(common_arg(
        {"-h", "--help", "--usage"},
        "print usage and exit",
        [](common_params & params) {
            params.usage = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD}));
    add_opt(common_arg(
        {"--version"},
        "show version and build info",
        [](common_params &) {
            llama_print_build_info(llama_version());
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"-cl", "--cache-list"},
        "show list of models in cache",
        [](common_params &) {
            auto models = common_list_cached_models();
            printf("number of models in cache: %zu\n", models.size());
            for (size_t i = 0; i < models.size(); i++) {
                printf("%4zu. %s\n", i + 1, models[i].to_string().c_str());
            }
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"--completion-bash"},
        "print source-able bash completion script for llama.cpp",
        [](common_params & params) {
            params.completion = true;
        }
    ));
    add_opt(common_arg(
        {"--server-base"}, "URL",
        string_format("connect to this server instead of starting a new one, example: 'http://localhost:8080' (default: none)"),
        [](common_params & params, const std::string & value) {
            params.server_base = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--verbose-prompt"},
        string_format("print a verbose prompt before generation (default: %s)", params.verbose_prompt ? "true" : "false"),
        [](common_params & params) {
            params.verbose_prompt = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--display-prompt"},
        {"--no-display-prompt"},
        string_format("whether to print prompt at generation (default: %s)", params.display_prompt ? "true" : "false"),
        [](common_params & params, bool value) {
            params.display_prompt = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-co", "--color"}, "[on|off|auto]",
        "Colorize output to distinguish prompt and user input from generations ('on', 'off', or 'auto', default: 'auto')\n"
        "'auto' enables colors when output is to a terminal",
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.use_color = true;
            } else if (is_falsey(value)) {
                params.use_color = false;
            } else if (is_autoy(value)) {
                params.use_color = tty_can_use_colors();
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --color: '%s'\n", value.c_str()));
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP}));
    add_opt(common_arg(
        {"-t", "--threads"}, "N",
        string_format("number of CPU threads to use during generation (default: %d)", params.cpuparams.n_threads),
        [](common_params & params, int value) {
            params.cpuparams.n_threads = value;
            if (params.cpuparams.n_threads <= 0) {
                params.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_env("LLAMA_ARG_THREADS"));
    add_opt(common_arg(
        {"-tb", "--threads-batch"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads)",
        [](common_params & params, int value) {
            params.cpuparams_batch.n_threads = value;
            if (params.cpuparams_batch.n_threads <= 0) {
                params.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ));
    add_opt(common_arg(
        {"-C", "--cpu-mask"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")",
        [](common_params & params, const std::string & mask) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Cr", "--cpu-range"}, "lo-hi",
        "range of CPUs for affinity. Complements --cpu-mask",
        [](common_params & params, const std::string & range) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(common_arg(
        {"--cpu-strict"}, "<0|1>",
        string_format("use strict CPU placement (default: %u)\n", (unsigned) params.cpuparams.strict_cpu),
        [](common_params & params, const std::string & value) {
            params.cpuparams.strict_cpu = std::stoul(value);
        }
    ));
    add_opt(common_arg(
        {"--prio"}, "N",
        string_format("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: %d)\n", params.cpuparams.priority),
        [](common_params & params, int prio) {
            if (prio < GGML_SCHED_PRIO_LOW || prio > GGML_SCHED_PRIO_REALTIME) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll"}, "<0...100>",
        string_format("use polling level to wait for work (0 - no polling, default: %u)\n", (unsigned) params.cpuparams.poll),
        [](common_params & params, const std::string & value) {
            params.cpuparams.poll = std::stoul(value);
        }
    ));
    add_opt(common_arg(
        {"-Cb", "--cpu-mask-batch"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Crb", "--cpu-range-batch"}, "lo-hi",
        "ranges of CPUs for affinity. Complements --cpu-mask-batch",
        [](common_params & params, const std::string & range) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(common_arg(
        {"--cpu-strict-batch"}, "<0|1>",
        "use strict CPU placement (default: same as --cpu-strict)",
        [](common_params & params, int value) {
            params.cpuparams_batch.strict_cpu = value;
        }
    ));
    add_opt(common_arg(
        {"--prio-batch"}, "N",
        string_format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.cpuparams_batch.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll-batch"}, "<0|1>",
        "use polling to wait for work (default: same as --poll)",
        [](common_params & params, int value) {
            params.cpuparams_batch.poll = value;
        }
    ));
    add_opt(common_arg(
        {"-lcs", "--lookup-cache-static"}, "FNAME",
        "path to static lookup cache to use for lookup decoding (not updated by generation)",
        [](common_params & params, const std::string & value) {
            params.speculative.ngram_cache.lookup_cache_static = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-lcd", "--lookup-cache-dynamic"}, "FNAME",
        "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
        [](common_params & params, const std::string & value) {
            params.speculative.ngram_cache.lookup_cache_dynamic = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-c", "--ctx-size"}, "N",
        string_format("size of the prompt context (default: %d, 0 = loaded from model)", params.n_ctx),
        [](common_params & params, int value) {
            params.n_ctx = value;
            if (value == 0) {
                // disable context reduction in llama_params_fit if the user explicitly requests the full context size:
                params.fit_params_min_ctx = UINT32_MAX;
            }
        }
    ).set_env("LLAMA_ARG_CTX_SIZE"));
    add_opt(common_arg(
        { "--kv-unified-per-slot" }, "N",
        "context limit per parallel slot (default: unset, behavior unchanged).\n"
        "when set without -c/--ctx-size, the shared KV pool is sized to n_parallel*N",
        [](common_params & params, int value) {
            params.kv_unified_per_slot = value;
        }
    ).set_env("LLAMA_ARG_KV_UNIFIED_PER_SLOT").set_examples({ LLAMA_EXAMPLE_SERVER }));
    add_opt(common_arg(
        {"-n", "--predict", "--n-predict"}, "N",
        string_format(
            ex == LLAMA_EXAMPLE_COMPLETION
                ? "number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)"
                : "number of tokens to predict (default: %d, -1 = infinity)",
            params.n_predict),
        [](common_params & params, int value) {
            params.n_predict = value;
        }
    ).set_env("LLAMA_ARG_N_PREDICT"));
    add_opt(common_arg(
        {"-b", "--batch-size"}, "N",
        string_format("logical maximum batch size (default: %d)", params.n_batch),
        [](common_params & params, int value) {
            params.n_batch = value;
        }
    ).set_env("LLAMA_ARG_BATCH"));
    add_opt(common_arg(
        {"-ub", "--ubatch-size"}, "N",
        string_format("physical maximum batch size (default: %d)", params.n_ubatch),
        [](common_params & params, int value) {
            params.n_ubatch = value;
        }
    ).set_env("LLAMA_ARG_UBATCH"));
    add_opt(common_arg(
        {"--keep"}, "N",
        string_format("number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n_keep),
        [](common_params & params, int value) {
            params.n_keep = value;
        }
    ));
    add_opt(common_arg(
        {"--swa-full"},
        string_format("use full-size SWA cache (default: %s)\n"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)", params.swa_full ? "true" : "false"),
        [](common_params & params) {
            params.swa_full = true;
        }
    ).set_env("LLAMA_ARG_SWA_FULL"));
    add_opt(common_arg(
        {"-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"}, "N",
        string_format("max number of context checkpoints to create per slot (default: %d)"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/15293)", params.n_ctx_checkpoints),
        [](common_params & params, int value) {
            params.n_ctx_checkpoints = value;
        }
    ).set_env("LLAMA_ARG_CTX_CHECKPOINTS").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-cms", "--checkpoint-min-step"}, "N",
        string_format("minimum spacing between context checkpoints in tokens (default: %d, 0 = no minimum)", params.checkpoint_min_step),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("checkpoint-min-step must be non-negative");
            }
            params.checkpoint_min_step = value;
        }
    ).set_env("LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-cram", "--cache-ram"}, "N",
        string_format("set the maximum cache size in MiB (default: %d, -1 - no limit, 0 - disable)"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/16391)", params.cache_ram_mib),
        [](common_params & params, int value) {
            params.cache_ram_mib = value;
        }
    ).set_env("LLAMA_ARG_CACHE_RAM").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-kvu", "--kv-unified"},
        {"-no-kvu", "--no-kv-unified"},
        "use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)",
        [](common_params & params, bool value) {
            params.kv_unified = value;
        }
    ).set_env("LLAMA_ARG_KV_UNIFIED").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_BATCHED, LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"--cache-idle-slots"},
        {"--no-cache-idle-slots"},
        "save idle slots to the prompt cache on new task, and clear them when using unified KV (default: enabled, requires cache-ram)",
        [](common_params & params, bool value) {
            params.cache_idle_slots = value;
        }
    ).set_env("LLAMA_ARG_CACHE_IDLE_SLOTS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--context-shift"},
        {"--no-context-shift"},
        string_format("whether to use context shift on infinite text generation (default: %s)", params.ctx_shift ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.ctx_shift = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY}).set_env("LLAMA_ARG_CONTEXT_SHIFT"));
    add_opt(common_arg(
        {"--chunks"}, "N",
        string_format("max number of chunks to process (default: %d, -1 = all)", params.n_chunks),
        [](common_params & params, int value) {
            params.n_chunks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg({ "-fa", "--flash-attn" }, "[on|off|auto]",
                       string_format("set Flash Attention use ('on', 'off', or 'auto', default: '%s')",
                                     llama_flash_attn_type_name(params.flash_attn_type)),
                       [](common_params & params, const std::string & value) {
                           if (is_truthy(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                           } else if (is_falsey(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
                           } else if (is_autoy(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
                           } else {
                               throw std::runtime_error(
                                   string_format("error: unknown value for --flash-attn: '%s'\n", value.c_str()));
                           }
                       }).set_env("LLAMA_ARG_FLASH_ATTN"));
    add_opt(common_arg(
        {"--prefetch-experts-slots"}, "N",
        "MoE expert H2D staging slots (default 0 = off; N>=2 enables full-tensor prefetch with 1-deep lookahead for host-resident/ncmoe-offloaded expert weights during prefill; recommended 3; capped at 4). GPU memory cost = N x max_expert_tensor.",
        [](common_params & params, const std::string & value) {
            params.prefetch_experts_slots = std::stoi(value);
        }
    ));
    add_opt(common_arg(
        {"-p", "--prompt"}, "PROMPT",
        "prompt to start generation with; for system message, use -sys",
        [](common_params & params, const std::string & value) {
            params.prompt = value;
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-sys", "--system-prompt"}, "PROMPT",
        "system prompt to use with model (if applicable, depending on chat template)",
        [](common_params & params, const std::string & value) {
            params.system_prompt = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_DIFFUSION, LLAMA_EXAMPLE_MTMD}));
    add_opt(common_arg(
        {"--perf"},
        {"--no-perf"},
        string_format("whether to enable internal libllama performance timings (default: %s)", params.no_perf ? "true" : "false"),
        [](common_params & params, bool value) {
            params.no_perf = !value;
            params.sampling.no_perf = !value;
        }
    ).set_env("LLAMA_ARG_PERF"));
    add_opt(common_arg(
        {"--show-timings"},
        {"--no-show-timings"},
        string_format("whether to show timing information after each response (default: %s)", params.show_timings ? "true" : "false"),
        [](common_params & params, bool value) {
            params.show_timings = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SHOW_TIMINGS"));
    add_opt(common_arg(
        {"-f", "--file"}, "FNAME",
        "a file containing the prompt (default: none)",
        [](common_params & params, const std::string & value) {
            params.prompt = read_file(value);
            // store the external file name in params
            params.prompt_file = value;
            if (!params.prompt.empty() && params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-sysf", "--system-prompt-file"}, "FNAME",
        "a file containing the system prompt (default: none)",
        [](common_params & params, const std::string & value) {
            params.system_prompt = read_file(value);
            if (!params.system_prompt.empty() && params.system_prompt.back() == '\n') {
                params.system_prompt.pop_back();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_DIFFUSION}));
    add_opt(common_arg(
        {"--in-file"}, "FNAME",
        "an input file (use comma-separated values to specify multiple files)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                std::ifstream file(item);
                if (!file) {
                    throw std::runtime_error(string_format("error: failed to open file '%s'\n", item.c_str()));
                }
                params.in_files.push_back(item);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"-bf", "--binary-file"}, "FNAME",
        "binary file containing the prompt (default: none)",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value, std::ios::binary);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            // store the external file name in params
            params.prompt_file = value;
            std::ostringstream ss;
            ss << file.rdbuf();
            params.prompt = ss.str();
            fprintf(stderr, "Read %zu bytes from binary file %s\n", params.prompt.size(), value.c_str());
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-e", "--escape"},
        {"--no-escape"},
        string_format("whether to process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)", params.escape ? "true" : "false"),
        [](common_params & params, bool value) {
            params.escape = value;
        }
    ));
    add_opt(common_arg(
        {"-ptc", "--print-token-count"}, "N",
        string_format("print token count every N tokens (default: %d)", params.n_print),
        [](common_params & params, int value) {
            params.n_print = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache"}, "FNAME",
        "file to cache prompt state for faster startup (default: none)",
        [](common_params & params, const std::string & value) {
            params.path_prompt_cache = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache-all"},
        "if specified, saves user input and generations to cache as well\n",
        [](common_params & params) {
            params.prompt_cache_all = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache-ro"},
        "if specified, uses the prompt cache but does not update it",
        [](common_params & params) {
            params.prompt_cache_ro = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-r", "--reverse-prompt"}, "PROMPT",
        "halt generation at PROMPT, return control in interactive mode\n",
        [](common_params & params, const std::string & value) {
            params.antiprompt.emplace_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-sp", "--special"},
        string_format("special tokens output enabled (default: %s)", params.special ? "true" : "false"),
        [](common_params & params) {
            params.special = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-cnv", "--conversation"},
        {"-no-cnv", "--no-conversation"},
        "whether to run in conversation mode:\n"
        "- does not print special tokens and suffix/prefix\n"
        "- interactive mode is also enabled\n"
        "(default: auto enabled if chat template is available)",
        [](common_params & params, bool value) {
            params.conversation_mode = value ? COMMON_CONVERSATION_MODE_ENABLED : COMMON_CONVERSATION_MODE_DISABLED;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-st", "--single-turn"},
        "run conversation for a single turn only, then exit when done\n"
        "will not be interactive if first turn is predefined with --prompt\n"
        "(default: false)",
        [](common_params & params) {
            params.single_turn = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-i", "--interactive"},
        string_format("run in interactive mode (default: %s)", params.interactive ? "true" : "false"),
        [](common_params & params) {
            params.interactive = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-if", "--interactive-first"},
        string_format("run in interactive mode and wait for input right away (default: %s)", params.interactive_first ? "true" : "false"),
        [](common_params & params) {
            params.interactive_first = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-mli", "--multiline-input"},
        "allows you to write or paste multiple lines without ending each in '\\'",
        [](common_params & params) {
            params.multiline_input = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--in-prefix-bos"},
        "prefix BOS to user inputs, preceding the `--in-prefix` string",
        [](common_params & params) {
            params.input_prefix_bos = true;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--in-prefix"}, "STRING",
        "string to prefix user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_prefix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--in-suffix"}, "STRING",
        "string to suffix after user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_suffix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--warmup"},
        {"--no-warmup"},
        string_format("whether to perform warmup with an empty run (default: %s)", params.warmup ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.warmup = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--spm-infill"},
        string_format(
            "use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: %s)",
            params.spm_infill ? "enabled" : "disabled"
        ),
        [](common_params & params) {
            params.spm_infill = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--samplers"}, "SAMPLERS",
        string_format("samplers that will be used for generation in the order, separated by \';\'\n(default: %s)", sampler_type_names.c_str()),
        [](common_params & params, const std::string & value) {
            const auto sampler_names = string_split<std::string>(value, ';');
            params.sampling.samplers = common_sampler_types_from_names(sampler_names);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-s", "--seed"}, "SEED",
        string_format("RNG seed (default: %d, use random seed for %d)", params.sampling.seed, LLAMA_DEFAULT_SEED),
        [](common_params & params, const std::string & value) {
            params.sampling.seed = std::stoul(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--sampler-seq", "--sampling-seq"}, "SEQUENCE",
        string_format("simplified sequence for samplers that will be used (default: %s)", sampler_type_chars.c_str()),
        [](common_params & params, const std::string & value) {
            params.sampling.samplers = common_sampler_types_from_chars(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--ignore-eos"},
        "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
        [](common_params & params) {
            params.sampling.ignore_eos = true;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--temp", "--temperature"}, "N",
        string_format("temperature (default: %.2f)", (double)params.sampling.temp),
        [](common_params & params, const std::string & value) {
            params.sampling.temp = std::stof(value);
            params.sampling.temp = std::max(params.sampling.temp, 0.0f);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
        }
    ).set_sampling().set_env("LLAMA_ARG_TEMPERATURE"));
    add_opt(common_arg(
        {"--top-k"}, "N",
        string_format("top-k sampling (default: %d, 0 = disabled)", params.sampling.top_k),
        [](common_params & params, int value) {
            params.sampling.top_k = value;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
        }
    ).set_sampling().set_env("LLAMA_ARG_TOP_K"));
    add_opt(common_arg(
        {"--top-p"}, "N",
        string_format("top-p sampling (default: %.2f, 1.0 = disabled)", (double)params.sampling.top_p),
        [](common_params & params, const std::string & value) {
            params.sampling.top_p = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P;
        }
    ).set_sampling().set_env("LLAMA_ARG_TOP_P"));
    add_opt(common_arg(
        {"--min-p"}, "N",
        string_format("min-p sampling (default: %.2f, 0.0 = disabled)", (double)params.sampling.min_p),
        [](common_params & params, const std::string & value) {
            params.sampling.min_p = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P;
        }
    ).set_sampling().set_env("LLAMA_ARG_MIN_P"));
    add_opt(common_arg(
        {"--top-nsigma", "--top-n-sigma"}, "N",
        string_format("top-n-sigma sampling (default: %.2f, -1.0 = disabled)", params.sampling.top_n_sigma),
        [](common_params & params, const std::string & value) {
            params.sampling.top_n_sigma = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--xtc-probability"}, "N",
        string_format("xtc probability (default: %.2f, 0.0 = disabled)", (double)params.sampling.xtc_probability),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_probability = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--xtc-threshold"}, "N",
        string_format("xtc threshold (default: %.2f, 1.0 = disabled)", (double)params.sampling.xtc_threshold),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_threshold = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--typical", "--typical-p"}, "N",
        string_format("locally typical sampling, parameter p (default: %.2f, 1.0 = disabled)", (double)params.sampling.typ_p),
        [](common_params & params, const std::string & value) {
            params.sampling.typ_p = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--repeat-last-n"}, "N",
        string_format("last n tokens to consider for penalize (default: %d, 0 = disabled)", params.sampling.penalty_last_n),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::runtime_error(string_format("error: invalid repeat-last-n = %d\n", value));
            }
            params.sampling.penalty_last_n = value;
            params.sampling.n_prev = std::max(params.sampling.n_prev, params.sampling.penalty_last_n);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--repeat-penalty"}, "N",
        string_format("penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)", (double)params.sampling.penalty_repeat),
        [](common_params & params, const std::string & value) {
            const float penalty_repeat = std::stof(value);
            if (!std::isfinite(penalty_repeat) ||
                penalty_repeat <= 0.0f ||
                !std::isfinite(1.0f/penalty_repeat)) {
                throw std::runtime_error("error: repeat-penalty must be finite and greater than 0\n");
            }
            params.sampling.penalty_repeat = penalty_repeat;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT;
        }
    ).set_sampling().set_env("LLAMA_ARG_REPEAT_PENALTY"));
    add_opt(common_arg(
        {"--presence-penalty"}, "N",
        string_format("repeat alpha presence penalty (default: %.2f, 0.0 = disabled)", (double)params.sampling.penalty_present),
        [](common_params & params, const std::string & value) {
            const float penalty_present = std::stof(value);
            if (!std::isfinite(penalty_present)) {
                throw std::runtime_error("error: presence-penalty must be finite\n");
            }
            params.sampling.penalty_present = penalty_present;
        }
    ).set_sampling().set_env("LLAMA_ARG_PRESENCE_PENALTY"));
    add_opt(common_arg(
        {"--frequency-penalty"}, "N",
        string_format("repeat alpha frequency penalty (default: %.2f, 0.0 = disabled)", (double)params.sampling.penalty_freq),
        [](common_params & params, const std::string & value) {
            const float penalty_freq = std::stof(value);
            if (!std::isfinite(penalty_freq)) {
                throw std::runtime_error("error: frequency-penalty must be finite\n");
            }
            params.sampling.penalty_freq = penalty_freq;
        }
    ).set_sampling().set_env("LLAMA_ARG_FREQUENCY_PENALTY"));
    add_opt(common_arg(
        {"--dry-multiplier"}, "N",
        string_format("set DRY sampling multiplier (default: %.2f, 0.0 = disabled)", (double)params.sampling.dry_multiplier),
        [](common_params & params, const std::string & value) {
            params.sampling.dry_multiplier = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-base"}, "N",
        string_format("set DRY sampling base value (default: %.2f)", (double)params.sampling.dry_base),
        [](common_params & params, const std::string & value) {
            float potential_base = std::stof(value);
            if (potential_base >= 1.0f)
            {
                params.sampling.dry_base = potential_base;
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-allowed-length"}, "N",
        string_format("set allowed length for DRY sampling (default: %d)", params.sampling.dry_allowed_length),
        [](common_params & params, int value) {
            params.sampling.dry_allowed_length = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-penalty-last-n"}, "N",
        string_format("set DRY penalty for the last n tokens (default: %d, 0 = disable)", params.sampling.dry_penalty_last_n),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::runtime_error(string_format("error: invalid dry-penalty-last-n = %d\n", value));
            }
            params.sampling.dry_penalty_last_n = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-sequence-breaker"}, "STRING",
        string_format("add sequence breaker for DRY sampling, clearing out default breakers (%s) in the process; use \"none\" to not use any sequence breakers\n",
            params.sampling.dry_sequence_breakers.empty() ? "none" :
            std::accumulate(std::next(params.sampling.dry_sequence_breakers.begin()),
                params.sampling.dry_sequence_breakers.end(),
                std::string("'") + (params.sampling.dry_sequence_breakers[0] == "\n" ? "\\n" : params.sampling.dry_sequence_breakers[0]) + "'",
                [](const std::string& a, const std::string& b) {
                    std::string formatted_b = (b == "\n") ? "\\n" : b;
                    return a + ", '" + formatted_b + "'";
                }).c_str()),
        [](common_params & params, const std::string & value) {
            static bool defaults_cleared = false;

            if (!defaults_cleared) {
                params.sampling.dry_sequence_breakers.clear();
                defaults_cleared = true;
            }

            if (value == "none") {
                params.sampling.dry_sequence_breakers.clear();
            } else {
                params.sampling.dry_sequence_breakers.emplace_back(value);
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--adaptive-target"}, "N",
        string_format("adaptive-p: select tokens near this probability (valid range 0.0 "
                      "to 1.0; negative = disabled) (default: %.2f)\n"
                      "[(more info)](https://github.com/ggml-org/llama.cpp/pull/17927)",
                      (double)params.sampling.adaptive_target),
        [](common_params & params, const std::string & value) {
            params.sampling.adaptive_target = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--adaptive-decay"}, "N",
        string_format("adaptive-p: decay rate for target adaptation over time. lower values "
                      "are more reactive, higher values are more stable.\n"
                      "(valid range 0.0 to 0.99) (default: %.2f)",
                      (double)params.sampling.adaptive_decay),
        [](common_params & params, const std::string & value) {
            params.sampling.adaptive_decay = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dynatemp-range"}, "N",
        string_format("dynamic temperature range (default: %.2f, 0.0 = disabled)", (double)params.sampling.dynatemp_range),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_range = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dynatemp-exp"}, "N",
        string_format("dynamic temperature exponent (default: %.2f)", (double)params.sampling.dynatemp_exponent),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_exponent = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat"}, "N",
        string_format("use Mirostat sampling.\nTop K, Nucleus and Locally Typical samplers are ignored if used.\n"
        "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", params.sampling.mirostat),
        [](common_params & params, int value) {
            params.sampling.mirostat = value;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat-lr"}, "N",
        string_format("Mirostat learning rate, parameter eta (default: %.2f)", (double)params.sampling.mirostat_eta),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_eta = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat-ent"}, "N",
        string_format("Mirostat target entropy, parameter tau (default: %.2f)", (double)params.sampling.mirostat_tau),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_tau = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-l", "--logit-bias"}, "TOKEN_ID(+/-)BIAS",
        "modifies the likelihood of token appearing in the completion,\n"
        "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
        "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'",
        [](common_params & params, const std::string & value) {
            std::stringstream ss(value);
            llama_token key;
            char sign;
            std::string value_str;
            try {
                if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                    const float bias = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                    params.sampling.logit_bias.push_back({key, bias});
                } else {
                    throw std::invalid_argument("invalid input format");
                }
            } catch (const std::exception&) {
                throw std::invalid_argument("invalid input format");
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--grammar"}, "GRAMMAR",
        "BNF-like grammar to constrain generations (see samples in grammars/ dir)",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, value};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--grammar-file"}, "FNAME",
        "file to read grammar from",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, read_file(value)};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-j", "--json-schema"}, "SCHEMA",
        "JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(json::parse(value))};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-jf", "--json-schema-file"}, "FILE",
        "File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string schema;
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(schema)
            );
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(json::parse(schema))};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-bs", "--backend-sampling"},
        "enable backend sampling (experimental) (default: disabled)",
        [](common_params & params) {
            params.sampling.backend_sampling = true;
        }
    ).set_sampling().set_env("LLAMA_ARG_BACKEND_SAMPLING"));
    add_opt(common_arg(
        {"--pooling"}, "{none,mean,cls,last,rank}",
        "pooling type for embeddings, use model default if unspecified",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "none") { params.pooling_type = LLAMA_POOLING_TYPE_NONE; }
            else if (value == "mean") { params.pooling_type = LLAMA_POOLING_TYPE_MEAN; }
            else if (value == "cls")  { params.pooling_type = LLAMA_POOLING_TYPE_CLS;  }
            else if (value == "last") { params.pooling_type = LLAMA_POOLING_TYPE_LAST; }
            else if (value == "rank") { params.pooling_type = LLAMA_POOLING_TYPE_RANK; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_DEBUG}).set_env("LLAMA_ARG_POOLING"));
    add_opt(common_arg(
        {"--attention"}, "{causal,non-causal}",
        "attention type for embeddings, use model default if unspecified",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "causal") { params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL; }
            else if (value == "non-causal") { params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--rope-scaling"}, "{none,linear,yarn}",
        "RoPE frequency scaling method, defaults to linear unless specified by the model",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "none") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
            else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
            else if (value == "yarn") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_ROPE_SCALING_TYPE"));
    add_opt(common_arg(
        {"--rope-scale"}, "N",
        "RoPE context scaling factor, expands context by a factor of N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = 1.0f / std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_SCALE"));
    add_opt(common_arg(
        {"--rope-freq-base"}, "N",
        "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
        [](common_params & params, const std::string & value) {
            params.rope_freq_base = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_BASE"));
    add_opt(common_arg(
        {"--rope-freq-scale"}, "N",
        "RoPE frequency scaling factor, expands context by a factor of 1/N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_SCALE"));
    add_opt(common_arg(
        {"--yarn-orig-ctx"}, "N",
        string_format("YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx),
        [](common_params & params, int value) {
            params.yarn_orig_ctx = value;
        }
    ).set_env("LLAMA_ARG_YARN_ORIG_CTX"));
    add_opt(common_arg(
        {"--yarn-ext-factor"}, "N",
        string_format("YaRN: extrapolation mix factor (default: %.2f, 0.0 = full interpolation)", (double)params.yarn_ext_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_ext_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_EXT_FACTOR"));
    add_opt(common_arg(
        {"--yarn-attn-factor"}, "N",
        string_format("YaRN: scale sqrt(t) or attention magnitude (default: %.2f)", (double)params.yarn_attn_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_attn_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_ATTN_FACTOR"));
    add_opt(common_arg(
        {"--yarn-beta-slow"}, "N",
        string_format("YaRN: high correction dim or alpha (default: %.2f)", (double)params.yarn_beta_slow),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_slow = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_SLOW"));
    add_opt(common_arg(
        {"--yarn-beta-fast"}, "N",
        string_format("YaRN: low correction dim or beta (default: %.2f)", (double)params.yarn_beta_fast),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_fast = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_FAST"));
    add_opt(common_arg(
        {"-gan", "--grp-attn-n"}, "N",
        string_format("group-attention factor (default: %d)", params.grp_attn_n),
        [](common_params & params, int value) {
            params.grp_attn_n = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_N").set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_PASSKEY}));
    add_opt(common_arg(
        {"-gaw", "--grp-attn-w"}, "N",
        string_format("group-attention width (default: %d)", params.grp_attn_w),
        [](common_params & params, int value) {
            params.grp_attn_w = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_W").set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-kvo", "--kv-offload"},
        {"-nkvo", "--no-kv-offload"},
        string_format("whether to enable KV cache offloading (default: %s)", params.no_kv_offload ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_kv_offload = !value;
        }
    ).set_env("LLAMA_ARG_KV_OFFLOAD"));
    add_opt(common_arg(
        {"--repack"},
        {"-nr", "--no-repack"},
        string_format("whether to enable weight repacking (default: %s)", params.no_extra_bufts ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_extra_bufts = !value;
        }
    ).set_env("LLAMA_ARG_REPACK"));
    add_opt(common_arg(
        {"--no-host"},
        "bypass host buffer allowing extra buffers to be used",
        [](common_params & params) {
            params.no_host = true;
        }
    ).set_env("LLAMA_ARG_NO_HOST"));
    add_opt(common_arg(
        {"-ctk", "--cache-type-k"}, "TYPE",
        string_format(
            "KV cache data type for K\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.cache_type_k)
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_k = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_K"));
    add_opt(common_arg(
        {"-ctv", "--cache-type-v"}, "TYPE",
        string_format(
            "KV cache data type for V\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.cache_type_v)
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_v = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_V"));
    add_opt(common_arg(
        {"--hellaswag"},
        "compute HellaSwag score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.hellaswag = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--hellaswag-tasks"}, "N",
        string_format("number of tasks to use when computing the HellaSwag score (default: %zu)", params.hellaswag_tasks),
        [](common_params & params, int value) {
            params.hellaswag_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--winogrande"},
        "compute Winogrande score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.winogrande = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--winogrande-tasks"}, "N",
        string_format("number of tasks to use when computing the Winogrande score (default: %zu)", params.winogrande_tasks),
        [](common_params & params, int value) {
            params.winogrande_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--multiple-choice"},
        "compute multiple choice score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.multiple_choice = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--multiple-choice-tasks"}, "N",
        string_format("number of tasks to use when computing the multiple choice score (default: %zu)", params.multiple_choice_tasks),
        [](common_params & params, int value) {
            params.multiple_choice_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--kl-divergence"},
        "computes KL-divergence to logits provided via --kl-divergence-base",
        [](common_params & params) {
            params.kl_divergence = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--save-all-logits", "--kl-divergence-base"}, "FNAME",
        "set logits file",
        [](common_params & params, const std::string & value) {
            params.logits_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--ppl-stride"}, "N",
        string_format("stride for perplexity calculation (default: %d)", params.ppl_stride),
        [](common_params & params, int value) {
            params.ppl_stride = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--ppl-output-type"}, "<0|1>",
        string_format("output type for perplexity calculation (default: %d)", params.ppl_output_type),
        [](common_params & params, int value) {
            params.ppl_output_type = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"-dt", "--defrag-thold"}, "N",
        string_format("KV cache defragmentation threshold (DEPRECATED)"),
        [](common_params & params, const std::string & value) {
            GGML_UNUSED(params);
            GGML_UNUSED(value);
            LOG_WRN("DEPRECATED: --defrag-thold is deprecated and no longer necessary to specify\n");
        }
    ).set_env("LLAMA_ARG_DEFRAG_THOLD"));
    add_opt(common_arg(
        {"--moe-expert-cache"}, "N",
        string_format("GPU cache slots per host-resident MoE expert layer, 0 = disabled (default: %d)", params.n_moe_cache_slots),
        [](common_params & params, int value) {
            params.n_moe_cache_slots = value;
        }
    ).set_env("LLAMA_ARG_MOE_EXPERT_CACHE"));
    add_opt(common_arg(
        {"--moe-expert-cache-inserts"}, "N",
        string_format("max expert uploads per layer per decode step for the MoE expert cache (default: %d)", params.n_moe_cache_inserts),
        [](common_params & params, int value) {
            params.n_moe_cache_inserts = value;
        }
    ).set_env("LLAMA_ARG_MOE_EXPERT_CACHE_INSERTS"));
    if (ex == LLAMA_EXAMPLE_SERVER) {
        // this is to make sure this option appears in the server-specific section of the help message
        add_opt(common_arg(
            {"-np", "--parallel"}, "N",
            string_format("number of server slots (default: %d, -1 = auto)", params.n_parallel),
            [](common_params & params, int value) {
                if (value == 0) {
                    throw std::invalid_argument("error: invalid value for n_parallel\n");
                }
                params.n_parallel = value;
            }
        ).set_env("LLAMA_ARG_N_PARALLEL").set_examples({LLAMA_EXAMPLE_SERVER}));
    } else {
        add_opt(common_arg(
            {"-np", "--parallel"}, "N",
            string_format("number of parallel sequences to decode (default: %d)", params.n_parallel),
            [](common_params & params, int value) {
                params.n_parallel = value;
            }
        ).set_env("LLAMA_ARG_N_PARALLEL"));
    }
    add_opt(common_arg(
        {"-ns", "--sequences"}, "N",
        string_format("number of sequences to decode (default: %d)", params.n_sequences),
        [](common_params & params, int value) {
            params.n_sequences = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-cb", "--cont-batching"},
        {"-nocb", "--no-cont-batching"},
        string_format("whether to enable continuous batching (a.k.a dynamic batching) (default: %s)", params.cont_batching ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.cont_batching = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CONT_BATCHING"));
    add_opt(common_arg(
        {"-mm", "--mmproj"}, "FILE",
        "path to a multimodal projector file. see tools/mtmd/README.md\n"
        "note: if -hf is used, this argument can be omitted",
        [](common_params & params, const std::string & value) {
            params.mmproj.path = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ"));
    add_opt(common_arg(
        {"-mmu", "--mmproj-url"}, "URL",
        "URL to a multimodal projector file. see tools/mtmd/README.md",
        [](common_params & params, const std::string & value) {
            params.mmproj.url = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ_URL"));
    add_opt(common_arg(
        {"--mmproj-auto"},
        {"--no-mmproj", "--no-mmproj-auto"},
        string_format("whether to use multimodal projector file (if available), useful when using -hf (default: %s)", params.no_mmproj ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_mmproj = !value;
        }
    ).set_examples({LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_DOWNLOAD}).set_env("LLAMA_ARG_MMPROJ_AUTO"));
    add_opt(common_arg(
        {"--mmproj-offload"},
        {"--no-mmproj-offload"},
        string_format("whether to enable GPU offloading for multimodal projector (default: %s)", params.mmproj_use_gpu ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.mmproj_use_gpu = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ_OFFLOAD"));
    add_opt(common_arg(
        // note: "-mmdev" must sort after "--rpc" in the preset map, else RPC devices are not registered yet
        {"-mmdev", "--mmproj-device"}, "DEVICE",
        "device to use for multimodal projector (none = don't offload, default: follows --device)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            if (value == "none") {
                params.mmproj_use_gpu = false;
                params.mmproj_device  = nullptr;
                return;
            }
            auto devices = parse_device_list(value);
            // parse_device_list pushes nullptr at back so devices is length 2 for single device.
            if (devices.size() > 2) {
                throw std::invalid_argument("only one device may be specified for mmproj");
            }
            params.mmproj_use_gpu = true;
            params.mmproj_device  = devices.front();
        }
    ).set_examples(mmproj_examples).set_env("MTMD_BACKEND_DEVICE")); // no LLAMA_ARG_ prefix for backward compatibility reason
    add_opt(common_arg(
        {"--image", "--audio", "--video"}, "FILE",
        "path to an image, audio, or video file. use with multimodal models, use comma-separated values for multiple files\n",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                params.image.emplace_back(item);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--image-min-tokens"}, "N",
        "minimum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)",
        [](common_params & params, int value) {
            params.image_min_tokens = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_IMAGE_MIN_TOKENS"));
    add_opt(common_arg(
        {"--image-max-tokens"}, "N",
        "maximum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)",
        [](common_params & params, int value) {
            params.image_max_tokens = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_IMAGE_MAX_TOKENS"));
    add_opt(common_arg(
        {"--mtmd-batch-max-tokens"}, "N",
        string_format("maximum number of image tokens per batch when encoding images (default: %d)", params.mtmd_batch_max_tokens),
        [](common_params & params, int value) {
            params.mtmd_batch_max_tokens = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MTMD_BATCH_MAX_TOKENS"));
    add_opt(common_arg(
        {"--video-fps"}, "N",
        string_format("target video frame rate (default: %.1f)", params.video_fps),
        [](common_params & params, const std::string & value) {
            params.video_fps = std::stof(value);
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_VIDEO_FPS"));
    add_opt(common_arg(
        {"--video-timestamp-interval"}, "N",
        string_format("interval in milliseconds between text timestamps (default: %" PRId64 ")", params.video_timestamp_interval_ms),
        [](common_params & params, int value) {
            params.video_timestamp_interval_ms = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_VIDEO_TIMESTAMP_INTERVAL"));
    add_opt(common_arg(
        {"--video-ffmpeg-dir"}, "DIR",
        "path to the directory containing ffmpeg and ffprobe (default: search in PATH)",
        [](common_params & params, const std::string & value) {
            params.video_ffmpeg_bin_dir = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_VIDEO_FFMPEG_DIR"));
    if (params.is_gen_docs || llama_supports_rpc()) {
        add_opt(common_arg(
            {"--rpc"}, "SERVERS",
            "comma-separated list of RPC servers (host:port)",
            [](common_params & params, const std::string & value) {
                add_rpc_devices(value);
                GGML_UNUSED(params);
            }
        ).set_env("LLAMA_ARG_RPC"));
    }
    add_opt(common_arg(
        {"-lm", "--load-mode"}, "MODE",
        "model loading mode (default: auto)\n"
        "- auto: mmap, unless a device does not support it\n"
        "- none: no special loading mode\n"
        "- mmap: memory-map model (if mmap disabled, slower load but may reduce pageouts if not using mlock)\n"
        "- mlock: force system to keep model in RAM rather than swapping or compressing\n"
        "- mmap+mlock: mmap + force system to keep model in RAM rather than swapping or compressing\n"
        "- dio: use DirectIO if available\n",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "auto")       { params.load_mode = LLAMA_LOAD_MODE_AUTO;       }
            else if (value == "none")       { params.load_mode = LLAMA_LOAD_MODE_NONE;       }
            else if (value == "mmap")       { params.load_mode = LLAMA_LOAD_MODE_MMAP;       }
            else if (value == "mlock")      { params.load_mode = LLAMA_LOAD_MODE_MLOCK;      }
            else if (value == "mmap+mlock") { params.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK; }
            else if (value == "dio")        { params.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;  }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_LOAD_MODE"));
    add_opt(common_arg(
        {"-lzm", "--lazy-mode"}, "MODE",
        "on-demand reading of certain tensors, for example per-layer embeddings (default: auto)\n"
        "- on: read the rows of such tensors from disk on demand instead of keeping them resident (requires mmap)\n"
        "- auto: on, but only for tensors larger than 4 GiB\n"
        "- off: always keep them resident",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "on")   { params.lazy_mode = LLAMA_LAZY_MODE_ON;   }
            else if (value == "auto") { params.lazy_mode = LLAMA_LAZY_MODE_AUTO; }
            else if (value == "off")  { params.lazy_mode = LLAMA_LAZY_MODE_OFF;  }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_LAZY_MODE"));
    add_opt(common_arg(
        {"--numa"}, "TYPE",
        "attempt optimizations that help on some NUMA systems\n"
        "- distribute: spread execution evenly over all nodes\n"
        "- isolate: only spawn threads on CPUs on the node that execution started on\n"
        "- numactl: use the CPU map provided by numactl\n"
        "if run without this previously, it is recommended to drop the system page cache before using this\n"
        "see https://github.com/ggml-org/llama.cpp/issues/1437",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "distribute" || value == "") { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
            else if (value == "isolate") { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
            else if (value == "numactl") { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_NUMA"));
    add_opt(common_arg(
        {"-dev", "--device"}, "<dev1,dev2,..>",
        "comma-separated list of devices to use for offloading (none = don't offload)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            params.devices = parse_device_list(value);
        }
    ).set_env("LLAMA_ARG_DEVICE"));
    add_opt(common_arg(
        {"--list-devices"},
        "print list of available devices and exit",
        [](common_params &) {
            common_print_available_devices();
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"-ot", "--override-tensor"}, "<tensor name pattern>=<buffer type>,...",
        "override tensor buffer type", [](common_params & params, const std::string & value) {
            parse_tensor_buffer_overrides(value, params.tensor_buft_overrides);
        }
    ).set_env("LLAMA_ARG_OVERRIDE_TENSOR"));
    add_opt(common_arg(
        {"-cmoe", "--cpu-moe"},
        "keep all Mixture of Experts (MoE) weights in the CPU",
        [](common_params & params) {
            params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
        }
    ).set_env("LLAMA_ARG_CPU_MOE"));
    add_opt(common_arg(
        {"-ncmoe", "--n-cpu-moe"}, "N",
        "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            llm_add_n_cpu_ffn_overrides(value, LLM_FFN_EXPS_REGEX, params.tensor_buft_overrides);
        }
    ).set_env("LLAMA_ARG_N_CPU_MOE"));
    add_opt(common_arg(
        {"-ncffn", "--n-cpu-ffn"}, "N",
        "keep the dense FFN weights of the first N layers in the CPU\n"
        "(dense models; for MoE expert weights use --n-cpu-moe)",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            llm_add_n_cpu_ffn_overrides(value, LLM_FFN_DENSE_REGEX, params.tensor_buft_overrides);
        }
    ).set_env("LLAMA_ARG_N_CPU_FFN"));
    GGML_ASSERT(params.n_gpu_layers < 0); // string_format would need to be extended for a default >= 0
    add_opt(common_arg(
        {"-ngl", "--gpu-layers", "--n-gpu-layers"}, "N",
        string_format("max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)", params.n_gpu_layers == -1 ? "auto" : "all"),
        [](common_params & params, const std::string & value) {
            if (value == "auto") {
                params.n_gpu_layers = -1;
            } else if (value == "all") {
                params.n_gpu_layers = -2;
            } else {
                params.n_gpu_layers = std::stoi(value);
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: no usable GPU found, --gpu-layers option will be ignored\n");
                fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\n");
                fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
            }
        }
    ).set_env("LLAMA_ARG_N_GPU_LAYERS"));
    add_opt(common_arg(
        {"-sm", "--split-mode"}, "{none,layer,row,tensor}",
        "how to split the model across multiple GPUs, one of:\n"
        "- none: use one GPU only\n"
        "- layer (default): split layers and KV across GPUs (pipelined)\n"
        "- row: split weight across GPUs by rows (parallelized)\n"
        "- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)",
        [](common_params & params, const std::string & value) {
            if (value == "none") {
                params.split_mode = LLAMA_SPLIT_MODE_NONE;
            } else if (value == "layer") {
                params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            } else if (value == "row") {
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
            } else if (value == "tensor") {
                params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
            } else {
                throw std::invalid_argument("invalid value");
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the split mode has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_SPLIT_MODE"));
    add_opt(common_arg(
        {"-ts", "--tensor-split"}, "N0,N1,N2,...",
        "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1",
        [](common_params & params, const std::string & value) {
            std::string arg_next = value;

            // split string by , and /
            const std::regex regex{ R"([,/]+)" };
            std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
            std::vector<std::string> split_arg{ it, {} };
            if (split_arg.size() >= llama_max_devices()) {
                throw std::invalid_argument(
                    string_format("got %zu input configs, but system only has %zu devices", split_arg.size(), llama_max_devices())
                );
            }
            for (size_t i = 0; i < llama_max_devices(); ++i) {
                if (i < split_arg.size()) {
                    params.tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    params.tensor_split[i] = 0.0f;
                }
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting a tensor split has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_TENSOR_SPLIT"));
    add_opt(common_arg(
        {"-mg", "--main-gpu"}, "INDEX",
        string_format("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: %d)", params.main_gpu),
        [](common_params & params, int value) {
            params.main_gpu = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the main GPU has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_MAIN_GPU"));
    add_opt(common_arg(
        { "-fit", "--fit" }, "[on|off]",
        string_format("whether to adjust unset arguments to fit in device memory ('on' or 'off', default: '%s')", params.fit_params ? "on" : "off"),
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.fit_params = true;
            } else if (is_falsey(value)) {
                params.fit_params = false;
            } else {
                throw std::runtime_error(
                    string_format("error: unknown value for --fit: '%s'\n", value.c_str()));
            }
        }
    ).set_env("LLAMA_ARG_FIT"));
    add_opt(common_arg(
        { "-fitp", "--fit-print" }, "[on|off]",
        string_format("print the estimated required memory ('on' or 'off', default: '%s')", params.fit_params_print ? "on" : "off"),
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.fit_params_print = true;
            } else if (is_falsey(value)) {
                params.fit_params_print = false;
            } else {
                throw std::runtime_error(
                    string_format("error: unknown value for --fit-print: '%s'\n", value.c_str()));
            }
        }
    ).set_examples({LLAMA_EXAMPLE_FIT_PARAMS}).set_env("LLAMA_ARG_FIT_ESTIMATE"));
    add_opt(common_arg(
        { "-fitt", "--fit-target" }, "MiB0,MiB1,MiB2,...",
        string_format("target margin per device for --fit, comma-separated list of values, "
            "single value is broadcast across all devices, default: %zu", params.fit_params_target[0]/(1024*1024)),
        [](common_params & params, const std::string & value) {
            std::string arg_next = value;

            // split string by , and /
            const std::regex regex{ R"([,/]+)" };
            std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
            std::vector<std::string> split_arg{ it, {} };
            if (split_arg.size() >= llama_max_devices()) {
                throw std::invalid_argument(
                    string_format("got %zu input configs, but system only has %zu devices", split_arg.size(), llama_max_devices())
                );
            }
            if (split_arg.size() == 1) {
                std::fill(params.fit_params_target.begin(), params.fit_params_target.end(), std::stoull(split_arg[0]) * 1024*1024);
                return;
            }
            for (size_t i = 0; i < split_arg.size(); i++) {
                params.fit_params_target[i] = std::stoull(split_arg[i]) * 1024*1024;
            }
        }
    ).set_env("LLAMA_ARG_FIT_TARGET"));
    add_opt(common_arg(
        { "-fitc", "--fit-ctx" }, "N",
        string_format("minimum ctx size that can be set by --fit option, default: %" PRIu32, params.fit_params_min_ctx),
        [](common_params & params, int value) {
            params.fit_params_min_ctx = value;
        }
    ).set_env("LLAMA_ARG_FIT_CTX"));
    add_opt(common_arg(
        {"--check-tensors"},
        string_format("check model tensor data for invalid values (default: %s)", params.check_tensors ? "true" : "false"),
        [](common_params & params) {
            params.check_tensors = true;
        }
    ));
    add_opt(common_arg(
        {"--override-kv"}, "KEY=TYPE:VALUE,...",
        "advanced option to override model metadata by key. to specify multiple overrides, either use comma-separated values.\n"
        "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false,tokenizer.ggml.add_eos_token=bool:false",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                if (!string_parse_kv_override(item.c_str(), params.kv_overrides)) {
                    throw std::runtime_error(string_format("error: Invalid type for KV override: %s\n", item.c_str()));
                }
            }
        }
    ));
    add_opt(common_arg(
        {"--op-offload"},
        {"--no-op-offload"},
        string_format("whether to offload host tensor operations to device (default: %s)", params.no_op_offload ? "false" : "true"),
        [](common_params & params, bool value) {
            params.no_op_offload = !value;
        }
    ));
    add_opt(common_arg(
        {"--lora"}, "FNAME",
        "path to LoRA adapter (use comma-separated values to load multiple adapters)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                params.lora_adapters.push_back({ item, 1.0, "", "", nullptr });
            }
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--lora-scaled"}, "FNAME:SCALE,...",
        "path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)\n"
        "note: use comma-separated values",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                auto parts = string_split<std::string>(item, ':');
                if (parts.size() != 2) {
                    throw std::invalid_argument("lora-scaled format: FNAME:SCALE");
                }
                params.lora_adapters.push_back({ parts[0], std::stof(parts[1]), "", "", nullptr });
            }
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--control-vector"}, "FNAME",
        "add a control vector\nnote: use comma-separated values to add multiple control vectors",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                params.control_vectors.push_back({ 1.0f, item, });
            }
        }
    ));
    add_opt(common_arg(
        {"--control-vector-scaled"}, "FNAME:SCALE,...",
        "add a control vector with user defined scaling SCALE\n"
        "note: use comma-separated values (format: FNAME:SCALE,...)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                auto parts = string_split<std::string>(item, ':');
                if (parts.size() != 2) {
                    throw std::invalid_argument("control-vector-scaled format: FNAME:SCALE");
                }
                params.control_vectors.push_back({ std::stof(parts[1]), parts[0] });
            }
        }
    ));
    add_opt(common_arg(
        {"--control-vector-layer-range"}, "START", "END",
        "layer range to apply the control vector(s) to, start and end inclusive",
        [](common_params & params, const std::string & start, const std::string & end) {
            params.control_vector_layer_start = std::stoi(start);
            params.control_vector_layer_end = std::stoi(end);
        }
    ));
    add_opt(common_arg(
        {"-a", "--alias"}, "STRING",
        "set model name aliases, comma-separated (to be used by API)",
        [](common_params & params, const std::string & value) {
            for (auto & alias : string_split<std::string>(value, ',')) {
                alias = string_strip(alias);
                if (!alias.empty()) {
                    params.model_alias.insert(alias);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ALIAS"));
    add_opt(common_arg(
        {"--tags"}, "STRING",
        "set model tags, comma-separated (informational, not used for routing)",
        [](common_params & params, const std::string & value) {
            for (auto & tag : string_split<std::string>(value, ',')) {
                tag = string_strip(tag);
                if (!tag.empty()) {
                    params.model_tags.insert(tag);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TAGS"));
    add_opt(common_arg(
        {"-m", "--model"}, "FNAME",
        ex == LLAMA_EXAMPLE_EXPORT_LORA
            ? "model path from which to load base model"
            : "model path to load",
        [](common_params & params, const std::string & value) {
            params.model.path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_MODEL"));
    add_opt(common_arg(
        {"-mu", "--model-url"}, "MODEL_URL",
        "model download url (default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.url = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_MODEL_URL"));
    add_opt(common_arg(
        { "-dr", "--docker-repo" }, "[<repo>/]<model>[:quant]",
        "Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest.\n"
        "example: gemma3\n"
        "(default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.docker_repo = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_DOCKER_REPO"));
    add_opt(common_arg(
        {"-hf", "-hfr", "--hf-repo"}, "<user>/<model>[:quant]",
        "Hugging Face model repository; quant is optional, case-insensitive, default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.\n"
        "mmproj is also downloaded automatically if available. to disable, add --no-mmproj\n"
        "example: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M\n"
        "(default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.hf_repo = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_HF_REPO"));
    add_opt(common_arg(
        {"-hff", "--hf-file"}, "FILE",
        "Hugging Face model file. If specified, it will override the quant in --hf-repo (default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.hf_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_HF_FILE"));
    add_opt(common_arg(
        {"-hft", "--hf-token"}, "TOKEN",
        "Hugging Face access token (default: value from HF_TOKEN environment variable)",
        [](common_params & params, const std::string & value) {
            params.hf_token = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("HF_TOKEN"));
    add_opt(common_arg(
        {"--mtp"},
        "also download the multi-token prediction (MTP) head, if available (default: unused)",
        [](common_params & params) {
            params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
        }
    ).set_examples({LLAMA_EXAMPLE_DOWNLOAD}));
    add_opt(common_arg(
        {"--dflash"},
        "also download the DFlash sidecar, if available (default: unused)",
        [](common_params & params) {
            params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
        }
    ).set_examples({LLAMA_EXAMPLE_DOWNLOAD}));
    add_opt(common_arg(
        {"--eagle3"},
        "also download the Eagle3 sidecar, if available (default: unused)",
        [](common_params & params) {
            params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3);
        }
    ).set_examples({LLAMA_EXAMPLE_DOWNLOAD}));
    add_opt(common_arg(
        {"--context-file"}, "FNAME",
        "file to load context from (use comma-separated values to specify multiple files)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                std::ifstream file(item, std::ios::binary);
                if (!file) {
                    throw std::runtime_error(string_format("error: failed to open file '%s'\n", item.c_str()));
                }
                params.context_files.push_back(item);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--chunk-size"}, "N",
        string_format("minimum length of embedded text chunks (default: %d)", params.chunk_size),
        [](common_params & params, int value) {
            params.chunk_size = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--chunk-separator"}, "STRING",
        string_format("separator between chunks (default: '%s')", params.chunk_separator.c_str()),
        [](common_params & params, const std::string & value) {
            params.chunk_separator = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--junk"}, "N",
        string_format("number of times to repeat the junk text (default: %d)", params.n_junk),
        [](common_params & params, int value) {
            params.n_junk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"--pos"}, "N",
        string_format("position of the passkey in the junk text (default: %d)", params.i_pos),
        [](common_params & params, int value) {
            params.i_pos = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY}));
    add_opt(common_arg(
        {"-o", "--output", "--output-file"}, "FNAME",
        string_format("output file (default: '%s')", params.out_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.out_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_CVECTOR_GENERATOR, LLAMA_EXAMPLE_EXPORT_LORA, LLAMA_EXAMPLE_TTS, LLAMA_EXAMPLE_FINETUNE,
                    LLAMA_EXAMPLE_RESULTS, LLAMA_EXAMPLE_EXPORT_GRAPH_OPS, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-ofreq", "--output-frequency"}, "N",
        string_format("output the imatrix every N iterations (default: %d)", params.n_out_freq),
        [](common_params & params, int value) {
            params.n_out_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--output-format"}, "{gguf,dat}",
        string_format("output format for imatrix file (default: %s)", params.imat_dat > 0 ? "dat" : "gguf"),
        [](common_params & params, const std::string & value) {
            /**/ if (value == "gguf") { params.imat_dat = -1; }
            else if (value == "dat")  { params.imat_dat = 1;  }
            else { throw std::invalid_argument("invalid output format"); }
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--save-frequency"}, "N",
        string_format("save an imatrix copy every N iterations (default: %d)", params.n_save_freq),
        [](common_params & params, int value) {
            params.n_save_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--process-output"},
        string_format("collect data for the output tensor (default: %s)", params.process_output ? "true" : "false"),
        [](common_params & params) {
            params.process_output = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--ppl"},
        {"--no-ppl"},
        string_format("whether to compute perplexity (default: %s)", params.compute_ppl ? "true" : "false"),
        [](common_params & params, bool value) {
            params.compute_ppl = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--chunk", "--from-chunk"}, "N",
        string_format("start processing the input from chunk N (default: %d)", params.i_chunk),
        [](common_params & params, int value) {
            params.i_chunk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--show-statistics"},
        string_format("show imatrix statistics and then exit (default: %s)", params.show_statistics ? "true" : "false"),
        [](common_params & params) {
            params.show_statistics = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--parse-special"},
        string_format("parse special tokens (chat, tool, etc) (default: %s)", params.parse_special ? "true" : "false"),
        [](common_params & params) {
            params.parse_special = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--ids"},
        string_format("only print the token IDs, in a Python-parseable list form like [1, 2, 3] (default: %s)", params.tokenize_ids ? "true" : "false"),
        [](common_params & params) {
            params.tokenize_ids = true;
        }
    ).set_examples({LLAMA_EXAMPLE_TOKENIZE}));
    add_opt(common_arg(
        {"--stdin"},
        string_format("read the prompt from stdin (takes precedence over -f/--file and -p/--prompt) (default: %s)", params.tokenize_stdin ? "true" : "false"),
        [](common_params & params) {
            params.tokenize_stdin = true;
        }
    ).set_examples({LLAMA_EXAMPLE_TOKENIZE}));
    add_opt(common_arg(
        {"--no-bos"},
        string_format("do not add a BOS token to the prompt, even if the model normally uses one (default: %s)", params.tokenize_no_bos ? "true" : "false"),
        [](common_params & params) {
            params.tokenize_no_bos = true;
        }
    ).set_examples({LLAMA_EXAMPLE_TOKENIZE}));
    add_opt(common_arg(
        {"--no-parse-special"},
        string_format("do not parse special tokens (chat, tool, etc) (default: %s)", !params.parse_special ? "true" : "false"),
        [](common_params & params) {
            params.parse_special = false;
        }
    ).set_examples({LLAMA_EXAMPLE_TOKENIZE}));
    add_opt(common_arg(
        {"--show-count"},
        string_format("print the total number of tokens (default: %s)", params.tokenize_show_count ? "true" : "false"),
        [](common_params & params) {
            params.tokenize_show_count = true;
        }
    ).set_examples({LLAMA_EXAMPLE_TOKENIZE}));
    add_opt(common_arg(
        {"-pps"},
        string_format("is the prompt shared across parallel sequences (default: %s)", params.is_pp_shared ? "true" : "false"),
        [](common_params & params) {
            params.is_pp_shared = true;
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-tgs"},
        string_format("is the text generation separated across the different sequences (default: %s)", params.is_tg_separate ? "true" : "false"),
        [](common_params & params) {
            params.is_tg_separate = true;
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-npp"}, "n0,n1,...",
        "number of prompt tokens",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pp.insert(params.n_pp.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"-ntg"}, "n0,n1,...",
        "number of text generation tokens",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_tg.insert(params.n_tg.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"-npl"}, "n0,n1,...",
        "number of parallel prompts",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pl.insert(params.n_pl.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"--embd-normalize"}, "N",
        string_format("normalisation for embeddings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)", params.embd_normalize),
        [](common_params & params, int value) {
            params.embd_normalize = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--embd-output-format"}, "FORMAT",
        "empty = default, \"array\" = [[],[]...], \"json\" = openai style, \"json+\" = same \"json\" + cosine similarity matrix, \"raw\" = plain whitespace-delimited output (one embedding per line)",
        [](common_params & params, const std::string & value) {
            params.embd_out = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--embd-separator"}, "STRING",
        "separator of embeddings (default \\n) for example \"<#sep#>\"",
        [](common_params & params, const std::string & value) {
            params.embd_sep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--cls-separator"}, "STRING",
        "separator of classification sequences (default \\t) for example \"<#seq#>\"",
        [](common_params & params, const std::string & value) {
            params.cls_sep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--host"}, "HOST",
        string_format("IP addresses to listen on, comma-separated, or UNIX socket paths ending in .sock; with multiple TCP addresses, :: binds IPv6 only; overlapping addresses result in undefined behavior (default: %s)", params.hostnames[0].c_str()),
        [](common_params & params, const std::string & value) {
            params.hostnames.clear();
            for (auto & host : parse_csv_row(value)) {
                host = string_strip(host);
                if (!host.empty()) {
                    params.hostnames.push_back(host);
                }
            }
            if (params.hostnames.empty()) {
                throw std::invalid_argument("--host requires at least one address");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_HOST"));
    add_opt(common_arg(
        {"--port"}, "PORT",
        string_format("port to listen (default: %d)", params.port),
        [](common_params & params, int value) {
            params.port = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PORT"));
    add_opt(common_arg(
        {"--reuse-port"},
        string_format("allow multiple sockets to bind to the same port (default: %s)", params.reuse_port ? "enabled" : "disabled"),
        [](common_params & params) {
            params.reuse_port = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_REUSE_PORT"));
    add_opt(common_arg(
        {"--path"}, "PATH",
        string_format("path to serve static files from (default: %s)", params.public_path.c_str()),
        [](common_params & params, const std::string & value) {
            params.public_path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_STATIC_PATH"));
    add_opt(common_arg(
        {"--cors-origins"}, "ORIGINS",
        string_format(
            "comma-separated list of allowed origins for CORS (default: %s)\n"
            "if set to special value 'localhost', reflect the Origin header only if it is localhost",
        params.cors_origins.c_str()),
        [](common_params & params, const std::string & value) {
            params.cors_origins = value;
            params.cors_origins_explicit = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CORS_ORIGINS"));
    add_opt(common_arg(
        {"--cors-methods"}, "METHODS",
        string_format("comma-separated list of allowed methods for CORS (default: %s)", params.cors_methods.c_str()),
        [](common_params & params, const std::string & value) {
            params.cors_methods = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CORS_METHODS"));
    add_opt(common_arg(
        {"--cors-headers"}, "HEADERS",
        string_format("comma-separated list of allowed headers for CORS (default: %s)", params.cors_headers.c_str()),
        [](common_params & params, const std::string & value) {
            params.cors_headers = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CORS_HEADERS"));
    add_opt(common_arg(
        {"--cors-credentials"},
        {"--no-cors-credentials"},
        string_format(
            "whether to allow credentials for CORS (default: %s)\n"
            "note: if this is enabled and --cors-origins is set to * (default), the Origin header will be echoed back, and credentials will always be allowed",
        params.cors_credentials ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.cors_credentials = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CORS_CREDENTIALS"));
    add_opt(common_arg(
        {"--api-prefix"}, "PREFIX",
        string_format("prefix path the server serves from, without the trailing slash (default: %s)", params.api_prefix.c_str()),
        [](common_params & params, const std::string & value) {
            params.api_prefix = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_API_PREFIX"));
    add_opt(common_arg(
        {"--ui-config", "--webui-config"}, "JSON",
        "JSON that provides default UI settings (overrides UI defaults)",
        [](common_params & params, const std::string & value) {
            params.ui_config_json = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_UI_CONFIG"));
    add_opt(common_arg(
        {"--ui-config-file", "--webui-config-file"}, "PATH",
        "JSON file that provides default UI settings (overrides UI defaults)",
        [](common_params & params, const std::string & value) {
            params.ui_config_json = read_file(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_UI_CONFIG_FILE"));
    add_opt(common_arg(
        {"--ui-mcp-proxy", "--webui-mcp-proxy"},
        {"--no-ui-mcp-proxy", "--no-webui-mcp-proxy"},
        "experimental: whether to enable MCP CORS proxy - do not enable in untrusted environments (default: disabled)",
        [](common_params & params, bool value) {
            params.ui_mcp_proxy = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_UI_MCP_PROXY"));
    add_opt(common_arg(
        {"--tools"}, "TOOL1,TOOL2,...",
        "experimental: whether to enable built-in tools for AI agents - do not enable in untrusted environments (default: no tools)\n"
        "specify \"all\" to enable all tools\n"
        "available tools: read_file, file_glob_search, grep_search, exec_shell_command, write_file, edit_file, get_info\n"
        "note: for security reasons, this will limit --cors-origins to localhost by default",
        [](common_params & params, const std::string & value) {
            params.server_tools = parse_csv_row(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TOOLS"));
    add_opt(common_arg(
        {"--tools-runtime"}, "OPTION",
        "experimental: run tools in a separate runtime environment (default: none, use host environment)\n"
        "available options:\n"
        "  'docker:<image>', 'podman:<image>': spin up a new container and reuse it for all invocations, clean up on server exit\n"
        "  'docker-container:<id>', 'podman-container:<id>': use an existing container by ID, won't stop on server exit\n"
        "  'ssh:<target>': run tools on a remote POSIX host over SSH, key-based auth and a trusted host key are required\n",
        [](common_params & params, const std::string & value) {
            params.server_tools_runtime = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TOOLS_RUNTIME"));
    add_opt(common_arg(
        {"--mcp-servers-config"}, "PATH",
        "experimental: path to JSON file with MCP server definitions (Cursor-compatible format) - do not enable in untrusted environments (default: none)\n"
        "note: for security reasons, this will limit --cors-origins to localhost by default",
        [](common_params & params, const std::string & value) {
            params.mcp_servers_config = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MCP_SERVERS_CONFIG"));
    add_opt(common_arg(
        {"--mcp-servers-json"}, "JSON",
        "experimental: inline JSON with MCP server definitions (Cursor-compatible format) - do not enable in untrusted environments (default: none)\n"
        "note: for security reasons, this will limit --cors-origins to localhost by default",
        [](common_params & params, const std::string & value) {
            params.mcp_servers_json = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MCP_SERVERS_JSON"));
    add_opt(common_arg(
        {"-ag", "--agent"},
        {"-no-ag", "--no-agent"},
        "whether to enable CORS proxy and all built-in tools - do not enable in untrusted environments (default: disabled)\n"
        "note: for security reasons, this will limit --cors-origins to localhost by default",
        [](common_params & params, bool value) {
            if (value) {
                params.server_tools = {"all"};
                params.ui_mcp_proxy = true;
            } else {
                params.server_tools.clear();
                params.ui_mcp_proxy = false;
            }
            // note: do not modify cors_origins here, as the options are not evaluated in order (user may explicitly set --cors-origins before --agent)
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_AGENT"));
    add_opt(common_arg(
        {"--ui", "--webui"},
        {"--no-ui", "--no-webui"},
        string_format("whether to enable the Web UI (default: %s)", params.ui ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.ui = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_UI"));
    add_opt(common_arg(
        {"--embedding", "--embeddings"},
        string_format("restrict to only support embedding use case; use only with dedicated embedding models (default: %s)", params.embedding ? "enabled" : "disabled"),
        [](common_params & params) {
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_DEBUG}).set_env("LLAMA_ARG_EMBEDDINGS"));
    add_opt(common_arg(
        {"--rerank", "--reranking"},
        string_format("enable reranking endpoint on server (default: %s)", "disabled"),
        [](common_params & params) {
            params.embedding = true;
            params.pooling_type = LLAMA_POOLING_TYPE_RANK;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_RERANKING"));
    add_opt(common_arg(
        {"--api-key"}, "KEY",
        "API key to use for authentication, multiple keys can be provided as a comma-separated list (default: none)",
        [](common_params & params, const std::string & value) {
            for (const auto & key : parse_csv_row(value)) {
                if (!key.empty()) {
                    params.api_keys.push_back(key);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_API_KEY"));
    add_opt(common_arg(
        {"--api-key-file"}, "FNAME",
        "path to file containing API keys, one per line; lines starting with a hash are treated as comments (default: none)",
        [](common_params & params, const std::string & value) {
            std::ifstream key_file(value);
            if (!key_file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string key;
            while (std::getline(key_file, key)) {
                if (!key.empty() && key[0] != '#') {
                    params.api_keys.push_back(key);
                }
            }
            key_file.close();
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_API_KEY_FILE"));
    add_opt(common_arg(
        {"--ssl-key-file"}, "FNAME",
        "path to file a PEM-encoded SSL private key",
        [](common_params & params, const std::string & value) {
            params.ssl_file_key = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_KEY_FILE"));
    add_opt(common_arg(
        {"--ssl-cert-file"}, "FNAME",
        "path to file a PEM-encoded SSL certificate",
        [](common_params & params, const std::string & value) {
            params.ssl_file_cert = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_CERT_FILE"));
    add_opt(common_arg(
        {"--chat-template-kwargs"}, "STRING",
        "sets additional params for the json template parser, must be a valid json object string, e.g. '{\"key1\":\"value1\",\"key2\":\"value2\"}'",
        [](common_params & params, const std::string & value) {
            auto parsed = json::parse(value);
            for (const auto & item : parsed.items()) {
                if (item.key() == "enable_thinking") {
                    LOG_WRN("Setting 'enable_thinking' via --chat-template-kwargs is deprecated. "
                            "Use --reasoning on / --reasoning off instead.\n");
                }
                if (item.key() == "preserve_reasoning") {
                    LOG_WRN("Setting 'preserve_reasoning' via --chat-template-kwargs is deprecated. "
                            "Use --reasoning-preserve / --no-reasoning-preserve instead.\n");
                }
                params.default_template_kwargs[item.key()] = item.value().dump();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_CHAT_TEMPLATE_KWARGS"));
    add_opt(common_arg(
        {"-to", "--timeout"}, "N",
        string_format("server read/write timeout in seconds (default: %d)", params.timeout_read),
        [](common_params & params, int value) {
            params.timeout_read  = value;
            params.timeout_write = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TIMEOUT"));
    add_opt(common_arg(
        {"--sse-ping-interval"}, "N",
        string_format("server SSE ping interval in seconds (-1 = disabled, default: %d)", params.sse_ping_interval),
        [](common_params & params, int value) {
            params.sse_ping_interval = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSE_PING_INTERVAL"));
    add_opt(common_arg(
        {"--threads-http"}, "N",
        string_format("number of threads used to process HTTP requests (default: %d)", params.n_threads_http),
        [](common_params & params, int value) {
            params.n_threads_http = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_THREADS_HTTP"));
    add_opt(common_arg(
        {"--cache-prompt"},
        {"--no-cache-prompt"},
        string_format("whether to enable prompt caching (default: %s)", params.cache_prompt ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.cache_prompt = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CACHE_PROMPT"));
    add_opt(common_arg(
        {"--cache-reuse"}, "N",
        string_format(
            "min chunk size to attempt reusing from the cache via KV shifting, requires prompt caching to be enabled (default: %d)\n"
            "[(card)](https://ggml.ai/f0.png)", params.n_cache_reuse
        ),
        [](common_params & params, int value) {
            params.n_cache_reuse = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CACHE_REUSE"));
    add_opt(common_arg(
        {"--metrics"},
        string_format("enable prometheus compatible metrics endpoint (default: %s)", params.endpoint_metrics ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_metrics = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_METRICS"));
    add_opt(common_arg(
        {"--props"},
        string_format("enable changing global properties via POST /props (default: %s)", params.endpoint_props ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_props = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_PROPS"));
    add_opt(common_arg(
        {"--slots"},
        {"--no-slots"},
        string_format("expose slots monitoring endpoint (default: %s)", params.endpoint_slots ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.endpoint_slots = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_SLOTS"));
    add_opt(common_arg(
        {"--slot-save-path"}, "PATH",
        "path to save slot kv cache (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.slot_save_path = value;
            if (!fs_is_directory(params.slot_save_path)) {
                throw std::invalid_argument("not a directory: " + value);
            }
            // if doesn't end with DIRECTORY_SEPARATOR, add it
            if (!params.slot_save_path.empty() && params.slot_save_path[params.slot_save_path.size() - 1] != DIRECTORY_SEPARATOR) {
                params.slot_save_path += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--media-path"}, "PATH",
        "directory for loading local media files; files can be accessed via file:// URLs using relative paths (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.media_path = value;
            if (!fs_is_directory(params.media_path)) {
                throw std::invalid_argument("not a directory: " + value);
            }
            // if doesn't end with DIRECTORY_SEPARATOR, add it
            if (!params.media_path.empty() && params.media_path[params.media_path.size() - 1] != DIRECTORY_SEPARATOR) {
                params.media_path += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--models-dir"}, "PATH",
        "directory containing models for the router server (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.models_dir = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_DIR"));
    add_opt(common_arg(
        {"--models-preset"}, "PATH",
        "path to INI file containing model presets for the router server (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.models_preset = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_PRESET"));
    add_opt(common_arg(
        {"--models-max"}, "N",
        string_format("for router server, maximum number of models to load simultaneously (default: %d, 0 = unlimited)", params.models_max),
        [](common_params & params, int value) {
            params.models_max = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_MAX"));
    add_opt(common_arg(
        {"--models-autoload"},
        {"--no-models-autoload"},
        string_format("for router server, whether to automatically load models (default: %s)", params.models_autoload ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.models_autoload = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_AUTOLOAD"));
    add_opt(common_arg(
        {"--jinja"},
        {"--no-jinja"},
        string_format("whether to use jinja template engine for chat (default: %s)", params.use_jinja ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.use_jinja = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_MTMD}).set_env("LLAMA_ARG_JINJA"));
    add_opt(common_arg(
        {"--reasoning-format"}, "FORMAT",
        "controls whether thought tags are allowed and/or extracted from the response, and in which format they're returned; one of:\n"
        "- none: leaves thoughts unparsed in `message.content`\n"
        "- deepseek: puts thoughts in `message.reasoning_content`\n"
        "- deepseek-legacy: keeps `<think>` tags in `message.content` while also populating `message.reasoning_content`\n"
        "(default: auto)",
        [](common_params & params, const std::string & value) {
            params.reasoning_format = common_reasoning_format_from_name(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_THINK"));
    add_opt(common_arg(
        {"-rea", "--reasoning"}, "[on|off|auto]",
        "Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template))",
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.enable_reasoning = 1;
                params.default_template_kwargs["enable_thinking"] = "true";
            } else if (is_falsey(value)) {
                params.enable_reasoning = 0;
                params.default_template_kwargs["enable_thinking"] = "false";
            } else if (is_autoy(value)) {
                params.enable_reasoning = -1;
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --reasoning: '%s'\n", value.c_str()));
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_REASONING"));
    add_opt(common_arg(
        {"--reasoning-effort"}, "LEVEL",
        "reasoning effort level given to the chat template: 'default' to keep the template default,\n"
        "or a level such as 'minimal', 'low', 'medium', 'high', 'xhigh' or 'max' (default: default)",
        [](common_params & params, const std::string & value) {
            if (value == "default") {
                params.default_template_kwargs.erase("reasoning_effort");
            } else {
                params.default_template_kwargs["reasoning_effort"] = json(value).dump();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_REASONING_EFFORT"));
    add_opt(common_arg(
        {"--reasoning-budget"}, "N",
        "token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)",
        [](common_params & params, int value) {
            if (value < -1) { throw std::invalid_argument("invalid value"); }
            params.sampling.reasoning_budget_tokens = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_THINK_BUDGET"));
    add_opt(common_arg(
        {"--reasoning-budget-message"}, "MESSAGE",
        "message injected before the end-of-thinking tag when reasoning budget is exhausted (default: none)",
        [](common_params & params, const std::string & value) {
            params.sampling.reasoning_budget_message = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_THINK_BUDGET_MESSAGE"));
    add_opt(common_arg(
        {"--reasoning-preserve"},
        {"--no-reasoning-preserve"},
        "preserve reasoning trace in the full history, not just the last assistant message (default: enabled)\n"
        "compatible with certain templates having 'supports_preserve_reasoning' capability\n"
        "example: https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking",
        [](common_params & params, bool value) {
            if (value) {
                params.default_template_kwargs["preserve_reasoning"] = "true";
            } else {
                params.default_template_kwargs["preserve_reasoning"] = "false";
            }
            params.preserve_reasoning_specified = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_REASONING_PRESERVE"));
    add_opt(common_arg(
        {"--chat-template"}, "JINJA_TEMPLATE",
        string_format(
            "set custom jinja chat template (default: template taken from model's metadata)\n"
            "if suffix/prefix are specified, template will be disabled\n"
            "only commonly used templates are accepted (unless --jinja is set before this flag):\n"
            "list of built-in templates:\n%s", list_builtin_chat_templates().c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.chat_template = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_MTMD}).set_env("LLAMA_ARG_CHAT_TEMPLATE"));
    add_opt(common_arg(
        {"--chat-template-file"}, "JINJA_TEMPLATE_FILE",
        string_format(
            "set custom jinja chat template file (default: template taken from model's metadata)\n"
            "if suffix/prefix are specified, template will be disabled\n"
            "only commonly used templates are accepted (unless --jinja is set before this flag):\n"
            "list of built-in templates:\n%s", list_builtin_chat_templates().c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.chat_template = read_file(value);
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CHAT_TEMPLATE_FILE"));
    add_opt(common_arg(
        {"--skip-chat-parsing"},
        {"--no-skip-chat-parsing"},
        string_format(
            "force a pure content parser, even if a Jinja template is specified; model will output everything "
            "in the content section, including any reasoning and/or tool calls (default: disabled)"
        ),
        [](common_params & params, bool value) {
            params.force_pure_content_parser = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SKIP_CHAT_PARSING"));
    add_opt(common_arg(
        {"--prefill-assistant"},
        {"--no-prefill-assistant"},
        string_format(
            "whether to prefill the assistant's response if the last message is an assistant message (default: prefill enabled)\n"
            "when this flag is set, if the last message is an assistant message then it will be treated as a full message and not prefilled\n"
        ),
        [](common_params & params, bool value) {
            params.prefill_assistant = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PREFILL_ASSISTANT"));
    add_opt(common_arg(
        {"-sps", "--slot-prompt-similarity"}, "SIMILARITY",
        string_format("how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", params.slot_prompt_similarity),
        [](common_params & params, const std::string & value) {
            params.slot_prompt_similarity = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--lora-init-without-apply"},
        string_format("load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", params.lora_init_without_apply ? "enabled" : "disabled"),
        [](common_params & params) {
            params.lora_init_without_apply = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--sleep-idle-seconds"}, "SECONDS",
        string_format("number of seconds of idleness after which the server will sleep (default: %d; -1 = disabled)", params.sleep_idle_seconds),
        [](common_params & params, int value) {
            if (value == 0 || value < -1) {
                throw std::invalid_argument("invalid value: cannot be 0 or less than -1");
            }
            params.sleep_idle_seconds = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--simple-io"},
        "use basic IO for better compatibility in subprocesses and limited consoles",
        [](common_params & params) {
            params.simple_io = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--positive-file"}, "FNAME",
        string_format("positive prompts file, one prompt per line (default: '%s')", params.cvector_positive_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.cvector_positive_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--negative-file"}, "FNAME",
        string_format("negative prompts file, one prompt per line (default: '%s')", params.cvector_negative_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.cvector_negative_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--pca-batch"}, "N",
        string_format("batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)", params.n_pca_batch),
        [](common_params & params, int value) {
            params.n_pca_batch = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--pca-iter"}, "N",
        string_format("number of iterations used for PCA (default: %d)", params.n_pca_iterations),
        [](common_params & params, int value) {
            params.n_pca_iterations = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--method"}, "{pca, mean}",
        "dimensionality reduction method to be used (default: pca)",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "pca") { params.cvector_dimre_method = DIMRE_METHOD_PCA; }
            else if (value == "mean") { params.cvector_dimre_method = DIMRE_METHOD_MEAN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--output-format"}, "{md,jsonl}",
        "output format for batched-bench results (default: md)",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "jsonl") { params.batched_bench_output_jsonl = true; }
            else if (value == "md") { params.batched_bench_output_jsonl = false; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"--log-disable"},
        "Log disable",
        [](common_params &) {
            common_log_pause(common_log_main());
        }
    ));
    add_opt(common_arg(
        {"--log-file"}, "FNAME",
        "Log to file",
        [](common_params &, const std::string & value) {
            common_log_set_file(common_log_main(), value.c_str());
        }
    ).set_env("LLAMA_ARG_LOG_FILE"));
    add_opt(common_arg(
        {"--log-jsonl"},
        {"--no-log-jsonl"},
        "Log as JSONL (one JSON object per line) to stdout, this also disables colored logging (default: disabled)",
        [](common_params &, bool value) {
            common_log_set_jsonl(value);
        }
    ).set_env("LLAMA_ARG_LOG_JSONL"));
    add_opt(common_arg(
        {"--log-prompts-dir"}, "PATH",
        "Log prompts to directory (auto-created if not present; only used for debugging, default: disabled)",
        [](common_params & params, const std::string & value) {
            params.path_prompts_log_dir = value;
            std::error_code ec;
            std::filesystem::create_directories(value, ec);
            if (ec) {
                fprintf(stderr, "warning: failed to create prompts-log-dir '%s': %s\n", value.c_str(), ec.message().c_str());
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--log-colors"}, "[on|off|auto]",
        "Set colored logging ('on', 'off', or 'auto', default: 'auto')\n"
        "'auto' enables colors when output is to a terminal",
        [](common_params &, const std::string & value) {
            if (is_truthy(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_ENABLED);
            } else if (is_falsey(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_DISABLED);
            } else if (is_autoy(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_AUTO);
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --log-colors: '%s'\n", value.c_str()));
            }
        }
    ).set_env("LLAMA_ARG_LOG_COLORS"));
    add_opt(common_arg(
        {"-v", "--verbose", "--log-verbose"},
        "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
        [](common_params & params) {
            params.verbosity = INT_MAX;
            common_log_set_verbosity_thold(INT_MAX);
        }
    ));
    add_opt(common_arg(
        {"--offline"},
        "Offline mode: forces use of cache, prevents network access",
        [](common_params & params) {
            params.offline = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_OFFLINE"));
    add_opt(common_arg(
        {"-lv", "--verbosity", "--log-verbosity"}, "N",
        string_format("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\n"
            " - 0: generic output\n"
            " - 1: error\n"
            " - 2: warning\n"
            " - 3: info\n"
            " - 4: trace (more info)\n"
            " - 5: debug\n"
            "(default: %d)\n", params.verbosity),
        [](common_params & params, int value) {
            params.verbosity = value;
            common_log_set_verbosity_thold(value);
        }
    ).set_env("LLAMA_ARG_LOG_VERBOSITY"));
    add_opt(common_arg(
        {"--log-prefix"},
        {"--no-log-prefix"},
        "Enable prefix in log messages",
        [](common_params &, bool value) {
            common_log_set_prefix(common_log_main(), value);
        }
    ).set_env("LLAMA_ARG_LOG_PREFIX"));
    add_opt(common_arg(
        {"--log-timestamps"},
        {"--no-log-timestamps"},
        "Enable timestamps in log messages",
        [](common_params &, bool value) {
            common_log_set_timestamps(common_log_main(), value);
        }
    ).set_env("LLAMA_ARG_LOG_TIMESTAMPS"));

    //
    // speculative parameters
    //

    add_opt(common_arg(
        {"--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft"}, "<user>/<model>[:quant]",
        "Same as --hf-repo, but for the draft model (default: unused)",
        [](common_params & params, const std::string & value) {
            params.speculative.draft.mparams.hf_repo = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_HF_REPO"));
    add_opt(common_arg(
        {"--spec-draft-threads", "-td", "--threads-draft"}, "N",
        "number of threads to use during generation (default: same as --threads)",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams.n_threads = value;
            if (params.speculative.draft.cpuparams.n_threads <= 0) {
                params.speculative.draft.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-threads-batch", "-tbd", "--threads-batch-draft"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads-draft)",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams_batch.n_threads = value;
            if (params.speculative.draft.cpuparams_batch.n_threads <= 0) {
                params.speculative.draft.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-mask", "-Cd", "--cpu-mask-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.speculative.draft.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.speculative.draft.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-range", "-Crd", "--cpu-range-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft",
        [](common_params & params, const std::string & range) {
            params.speculative.draft.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.speculative.draft.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-strict", "--cpu-strict-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: same as --cpu-strict)",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams.strict_cpu = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-prio", "--prio-draft"}, "N",
        string_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.speculative.draft.cpuparams.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.draft.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-poll", "--poll-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: same as --poll)",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams.poll = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-mask-batch", "-Cbd", "--cpu-mask-batch-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.speculative.draft.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.speculative.draft.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-range-batch", "-Crbd", "--cpu-range-batch-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft-batch)",
        [](common_params & params, const std::string & range) {
            params.speculative.draft.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.speculative.draft.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--spec-draft-cpu-strict-batch", "--cpu-strict-batch-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: --cpu-strict-draft)",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams_batch.strict_cpu = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-prio-batch", "--prio-batch-draft"}, "N",
        string_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.speculative.draft.cpuparams_batch.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.draft.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-poll-batch", "--poll-batch-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: --poll-draft)",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams_batch.poll = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-type-k", "-ctkd", "--cache-type-k-draft"}, "TYPE",
        string_format(
            "KV cache data type for K for the draft model\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.speculative.draft.cache_type_k)
        ),
        [](common_params & params, const std::string & value) {
            params.speculative.draft.cache_type_k = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K"));
    add_opt(common_arg(
        {"--spec-draft-type-v", "-ctvd", "--cache-type-v-draft"}, "TYPE",
        string_format(
            "KV cache data type for V for the draft model\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.speculative.draft.cache_type_v)
        ),
        [](common_params & params, const std::string & value) {
            params.speculative.draft.cache_type_v = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V"));
    add_opt(common_arg(
        {"--spec-draft-override-tensor", "-otd", "--override-tensor-draft"}, "<tensor name pattern>=<buffer type>,...",
        "override tensor buffer type for draft model", [](common_params & params, const std::string & value) {
            parse_tensor_buffer_overrides(value, params.speculative.draft.tensor_buft_overrides);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-moe", "-cmoed", "--cpu-moe-draft"},
        "keep all Mixture of Experts (MoE) weights in the CPU for the draft model",
        [](common_params & params) {
            params.speculative.draft.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_CPU_MOE"));
    add_opt(common_arg(
        {"--spec-draft-n-cpu-moe", "--spec-draft-ncmoe", "-ncmoed", "--n-cpu-moe-draft"}, "N",
        "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU for the draft model",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            llm_add_n_cpu_ffn_overrides(value, LLM_FFN_EXPS_REGEX, params.speculative.draft.tensor_buft_overrides);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE"));

    add_opt(common_arg(
        {"--spec-draft-n-max"}, "N",
        string_format("number of tokens to draft for speculative decoding (default: %d)", params.speculative.draft.n_max),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.draft.n_max = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_N_MAX"));
    add_opt(common_arg(
        {"--spec-draft-n-min"}, "N",
        string_format("minimum number of draft tokens to use for speculative decoding (default: %d)", params.speculative.draft.n_min),
        [](common_params & params, int value) {
            params.speculative.draft.n_min = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_N_MIN"));
    add_opt(common_arg(
        {"--spec-synth-len"}, "L",
        "target mean synthetic acceptance length, including the target token (benchmarking only)",
        [](common_params & params, const std::string & value) {
            const std::string text = string_strip(value);
            size_t pos = 0;
            const double length = std::stod(text, &pos);
            if (pos != text.size() || length == -1.0) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.synth_len = length;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_SYNTH_LEN"));
    add_opt(common_arg(
        {"--spec-synth-rates"}, "P0,P1,...",
        "comma-separated unconditional per-position synthetic acceptance probabilities (benchmarking only)",
        [](common_params & params, const std::string & value) {
            const auto values = string_split<std::string>(value, ',');
            std::vector<double> rates;
            rates.reserve(values.size());
            for (const auto & raw : values) {
                const std::string text = string_strip(raw);
                size_t pos = 0;
                const double rate = std::stod(text, &pos);
                if (pos != text.size()) {
                    throw std::invalid_argument("invalid value");
                }
                rates.push_back(rate);
            }
            params.speculative.synth_rates = std::move(rates);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_SYNTH_RATES"));

    add_opt(common_arg(
        {"--spec-draft-p-split", "--draft-p-split"}, "P",
        string_format("speculative decoding split probability (default: %.2f)", (double)params.speculative.draft.p_split),
        [](common_params & params, const std::string & value) {
            params.speculative.draft.p_split = std::stof(value);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_P_SPLIT"));
    add_opt(common_arg(
        {"--spec-draft-p-min", "--draft-p-min"}, "P",
        string_format("minimum speculative decoding probability (greedy) (default: %.2f)", (double)params.speculative.draft.p_min),
        [](common_params & params, const std::string & value) {
            params.speculative.draft.p_min = std::stof(value);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_P_MIN"));
    add_opt(common_arg(
        {"--spec-draft-backend-sampling"},
        {"--no-spec-draft-backend-sampling"},
        string_format("offload draft sampling to the backend (default: %s)",
                      params.speculative.draft.backend_sampling ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.speculative.draft.backend_sampling = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING"));
    add_opt(common_arg(
        {"--spec-draft-device", "-devd", "--device-draft"}, "<dev1,dev2,..>",
        "comma-separated list of devices to use for offloading the draft model (none = don't offload, default: follows --device)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            params.speculative.draft.devices = parse_device_list(value);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    GGML_ASSERT(params.speculative.draft.n_gpu_layers < 0); // string_format would need to be extended for a default >= 0
    add_opt(common_arg(
        {"--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}, "N",
        string_format("max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)",
            params.speculative.draft.n_gpu_layers == -1 ? "auto" : "all"),
        [](common_params & params, const std::string & value) {
            if (value == "auto") {
                params.speculative.draft.n_gpu_layers = -1;
            } else if (value == "all") {
                params.speculative.draft.n_gpu_layers = -2;
            } else {
                params.speculative.draft.n_gpu_layers = std::stoi(value);
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: no usable GPU found, --gpu-layers-draft option will be ignored\n");
                fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\n");
                fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_N_GPU_LAYERS_DRAFT"));
    add_opt(common_arg(
        {"--spec-draft-model", "-md", "--model-draft"}, "FNAME",
        "draft model for speculative decoding (default: unused)",
        [](common_params & params, const std::string & value) {
            params.speculative.draft.mparams.path = value;
            params.speculative.draft.mparams.hf_file = value; // will be used if --spec-draft-hf is set
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_MODEL"));
    add_opt(common_arg(
        {"--spec-type"}, common_speculative_all_types_str(),
        string_format("comma-separated list of types of speculative decoding to use (default: %s)\n",
            common_speculative_type_name_str(params.speculative.types).c_str()),
        [](common_params & params, const std::string & value) {
            const auto types_str = string_split<std::string>(value, ',');
            auto types = common_speculative_types_from_names(types_str);
            params.speculative.types.insert(params.speculative.types.end(), types.begin(), types.end());
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_TYPE"));
    add_opt(common_arg(
        {"--spec-ngram-mod-n-min"}, "N",
        string_format("minimum number of ngram tokens to use for ngram-based speculative decoding (default: %d)", params.speculative.ngram_mod.n_min),
        [](common_params & params, int value) {
            if (value < 0 || value > 1024) {
                throw std::invalid_argument("ngram n-min must be between 0 and 1024 inclusive");
            }
            params.speculative.ngram_mod.n_min = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-mod-n-max"}, "N",
        string_format("maximum number of ngram tokens to use for ngram-based speculative decoding (default: %d)", params.speculative.ngram_mod.n_max),
        [](common_params & params, int value) {
            if (value < 0 || value > 1024) {
                throw std::invalid_argument("ngram n-max must be between 0 and 1024 inclusive");
            }
            params.speculative.ngram_mod.n_max = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-mod-n-match"}, "N",
        string_format("ngram-mod lookup length (default: %d)", params.speculative.ngram_mod.n_match),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_mod.n_match = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-ngram-simple-size-n"}, "N",
        string_format("ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram_simple.size_n),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_simple.size_n = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-simple-size-m"}, "N",
        string_format("ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram_simple.size_m),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_simple.size_m = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-simple-min-hits"}, "N",
        string_format("minimum hits for ngram-simple speculative decoding (default: %d)", params.speculative.ngram_simple.min_hits),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("ngram min hits must be at least 1");
            }
            params.speculative.ngram_simple.min_hits = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-ngram-map-k-size-n"}, "N",
        string_format("ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram_map_k.size_n),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_map_k.size_n = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k-size-m"}, "N",
        string_format("ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram_map_k.size_m),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_map_k.size_m = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k-min-hits"}, "N",
        string_format("minimum hits for ngram-map-k speculative decoding (default: %d)", params.speculative.ngram_map_k.min_hits),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("ngram min hits must be at least 1");
            }
            params.speculative.ngram_map_k.min_hits = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-ngram-map-k4v-size-n"}, "N",
        string_format("ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram_map_k4v.size_n),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_map_k4v.size_n = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k4v-size-m"}, "N",
        string_format("ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram_map_k4v.size_m),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_map_k4v.size_m = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k4v-min-hits"}, "N",
        string_format("minimum hits for ngram-map-k4v speculative decoding (default: %d)", params.speculative.ngram_map_k4v.min_hits),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("ngram min hits must be at least 1");
            }
            params.speculative.ngram_map_k4v.min_hits = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    //
    // removed params
    //

    add_opt(common_arg(
        {"--draft", "--draft-n", "--draft-max"}, "N",
        "the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use --spec-draft-n-max or --spec-ngram-mod-n-max");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_DRAFT_MAX"));
    add_opt(common_arg(
        {"--draft-min", "--draft-n-min"}, "N",
        "the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use --spec-draft-n-min or --spec-ngram-mod-n-min");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_DRAFT_MIN"));
    add_opt(common_arg(
        {"--spec-ngram-size-n"}, "N",
        "the argument has been removed. use the respective --spec-ngram-*-size-n or --spec-ngram-mod-n-match",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use the respective --spec-ngram-*-size-n");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--spec-ngram-size-m"}, "N",
        "the argument has been removed. use the respective --spec-ngram-*-size-m",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use the respective --spec-ngram-*-size-m");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--spec-ngram-min-hits"}, "N",
        "the argument has been removed. use the respective --spec-ngram-*-min-hits",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use the respective --spec-ngram-*-min-hits");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}));

    //
    // TTS params
    //

    add_opt(common_arg(
        {"--tts-lang"}, "FNAME",
        "language (ISO 639-1) for audio generation\n"
        "see tts/README.md for per-model usage notes",
        [](common_params & params, const std::string & value) {
            params.tts_lang = value;
        }
    ).set_examples({LLAMA_EXAMPLE_TTS}));
    add_opt(common_arg(
        {"--tts-speaker-file"}, "FNAME",
        "speaker file path for audio generation",
        [](common_params & params, const std::string & value) {
            params.tts_speaker_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_TTS}));

    //
    // diffusion params
    //

    add_opt(common_arg(
        {"--diffusion-steps"}, "N",
        string_format("number of diffusion steps (default: %d)", params.diffusion.steps),
        [](common_params & params, int value) { params.diffusion.steps = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-visual"},
        string_format("enable visual diffusion mode (show progressive generation) (default: %s)", params.diffusion.visual_mode ? "true" : "false"),
        [](common_params & params) { params.diffusion.visual_mode = true; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eps"}, "F",
        string_format("epsilon for timesteps (default: %.6f)", (double) params.diffusion.eps),
        [](common_params & params, const std::string & value) { params.diffusion.eps = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-algorithm"}, "N",
        string_format(
            "diffusion algorithm: 0=DIFFUSION_ALGORITHM_ORIGIN, 1=DIFFUSION_ALGORITHM_ENTROPY_BASED, "
            "2=DIFFUSION_ALGORITHM_MARGIN_BASED, 3=DIFFUSION_ALGORITHM_RANDOM, "
            "4=DIFFUSION_ALGORITHM_CONFIDENCE_BASED (default: %d)", params.diffusion.algorithm),
        [](common_params & params, int value) { params.diffusion.algorithm = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-alg-temp"}, "F",
        string_format("dream algorithm temperature (default: %.3f)", (double) params.diffusion.alg_temp),
        [](common_params & params, const std::string & value) { params.diffusion.alg_temp = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-block-length"}, "N",
        string_format("llada block length for generation (default: %d)", params.diffusion.block_length),
        [](common_params & params, int value) { params.diffusion.block_length = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-cfg-scale"}, "F",
        string_format("llada classifier-free guidance scale (default: %.3f)", (double) params.diffusion.cfg_scale),
        [](common_params & params, const std::string & value) { params.diffusion.cfg_scale = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-add-gumbel-noise"}, "F",
        string_format("add gumbel noise to the logits if temp > 0.0 (default: %s)", params.diffusion.add_gumbel_noise ? "true" : "false"),
        [](common_params & params, const std::string & value) { params.diffusion.add_gumbel_noise = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        { "-lr", "--learning-rate" }, "ALPHA",
        string_format("adamw or sgd optimizer alpha (default: %.2g); note: sgd alpha recommended ~10x (no momentum)", (double) params.lr.lr0),
        [](common_params & params, const std::string & value) { params.lr.lr0 = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg({ "-lr-min", "--learning-rate-min" }, "ALPHA",
        string_format("(if >0) final learning rate after decay (if -decay-epochs is set, default=%.2g)",
            (double) params.lr.lr_min),
        [](common_params & params, const std::string & value) { params.lr.lr_min = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-decay-epochs", "--learning-rate-decay-epochs"}, "ALPHA",
        string_format("(if >0) decay learning rate to -lr-min after this many epochs (exponential decay, default=%.2g)", (double) params.lr.decay_epochs),
        [](common_params & params, const std::string & value) { params.lr.decay_epochs = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-wd", "--weight-decay"}, "WD",
        string_format("adamw or sgd optimizer weight decay (0 is off; recommend very small e.g. 1e-9) (default: %.2g).", (double) params.lr.wd),
        [](common_params & params, const std::string & value) { params.lr.wd = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-val-split", "--val-split"}, "FRACTION",
        string_format("fraction of data to use as validation set for training (default: %.2g).", (double) params.val_split),
        [](common_params & params, const std::string & value) { params.val_split = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-epochs", "--epochs"}, "N",
        string_format("optimizer max # of epochs (default: %d)", params.lr.epochs),
        [](common_params & params, int epochs) { params.lr.epochs = epochs; }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-opt", "--optimizer"}, "sgd|adamw", "adamw or sgd",
        [](common_params & params, const std::string & name) {
            params.optimizer = common_opt_get_optimizer(name.c_str());
            if (params.optimizer == GGML_OPT_OPTIMIZER_TYPE_COUNT) {
                throw std::invalid_argument("invalid --optimizer, valid options: adamw, sgd");
            }
        }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"--check"},
        string_format("check rather than generate results (default: %s)", params.check ? "true" : "false"),
        [](common_params & params) {
            params.check = true;
        }
    ).set_examples({LLAMA_EXAMPLE_RESULTS}));
    add_opt(common_arg(
        {"--save-logits"},
        string_format("save final logits to files for verification (default: %s)", params.save_logits ? "true" : "false"),
        [](common_params & params) {
            params.save_logits = true;
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--logits-output-dir"}, "PATH",
        string_format("directory for saving logits output files (default: %s)", params.logits_output_dir.c_str()),
        [](common_params & params, const std::string & value) {
            params.logits_output_dir = value;
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--tensor-filter"}, "REGEX",
        "filter tensor names for debug output (regex pattern, can be specified multiple times)",
        [](common_params & params, const std::string & value) {
            params.tensor_filter.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));

    // presets

    add_opt(common_arg(
        {"--embd-gemma-default"},
        string_format("use default EmbeddingGemma model (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF";
            params.model.hf_file = "embeddinggemma-300M-qat-Q4_0.gguf";
            params.port = 8011;
            params.n_ubatch = 2048;
            params.n_batch = 2048;
            params.n_parallel = 32;
            params.n_ctx = 2048*params.n_parallel;
            params.verbose_prompt = true;
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-1.5b-default"},
        string_format("use default Qwen 2.5 Coder 1.5B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-1.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-3b-default"},
        string_format("use default Qwen 2.5 Coder 3B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-3b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-7b-default"},
        string_format("use default Qwen 2.5 Coder 7B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-7b-spec"},
        string_format("use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.speculative.draft.mparams.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.draft.mparams.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-14b-spec"},
        string_format("use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-14B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-14b-q8_0.gguf";
            params.speculative.draft.mparams.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.draft.mparams.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-30b-default"},
        string_format("use default Qwen 3 Coder 30B A3B Instruct (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8_0-GGUF";
            params.model.hf_file = "qwen3-coder-30b-a3b-instruct-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--gpt-oss-20b-default"},
        string_format("use gpt-oss-20b (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gpt-oss-20b-GGUF";
            params.model.hf_file = "gpt-oss-20b-mxfp4.gguf";
            params.port = 8013;
            params.n_ubatch = 2048;
            params.n_batch = 32768;
            params.n_parallel = 2;
            params.n_ctx = 131072*params.n_parallel;
            params.sampling.temp = 1.0f;
            params.sampling.top_p = 1.0f;
            params.sampling.top_k = 0;
            params.sampling.min_p = 0.01f;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--gpt-oss-120b-default"},
        string_format("use gpt-oss-120b (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gpt-oss-120b-GGUF";
            params.port = 8013;
            params.n_ubatch = 2048;
            params.n_batch = 32768;
            params.n_parallel = 2;
            params.n_ctx = 131072*params.n_parallel;
            params.sampling.temp = 1.0f;
            params.sampling.top_p = 1.0f;
            params.sampling.top_k = 0;
            params.sampling.min_p = 0.01f;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--vision-gemma-4b-default"},
        string_format("use Gemma 3 4B QAT (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gemma-3-4b-it-qat-GGUF";
            params.port = 8014;
            params.n_ctx = 0;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--vision-gemma-12b-default"},
        string_format("use Gemma 3 12B QAT (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gemma-3-12b-it-qat-GGUF";
            params.port = 8014;
            params.n_ctx = 0;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-default"},
        string_format("enable default speculative decoding config"),
        [](common_params & params) {
            params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
            params.speculative.ngram_mod.n_match = 24;
            params.speculative.ngram_mod.n_min = 48;
            params.speculative.ngram_mod.n_max = 64;

            // TODO: not sure if this is a good config - explore more settings and potentially enable it
            //params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
            //params.speculative.ngram_map_k4v.size_n = 8;
            //params.speculative.ngram_map_k4v.size_m = 24;
            //params.speculative.ngram_map_k4v.min_hits = 2;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    return ctx_arg;
}

void common_params_add_preset_options(std::vector<common_arg> & args) {
    // arguments below won't be treated as CLI args, only preset options
    args.push_back(common_arg(
        {"load-on-startup"}, "NAME",
        "in server router mode, autoload this model on startup",
        [](common_params &, const std::string &) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_LOAD_ON_STARTUP).set_preset_only());

    args.push_back(common_arg(
        {"stop-timeout"}, "SECONDS",
        "in server router mode, force-kill model instance after this many seconds of graceful shutdown",
        [](common_params &, int) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_STOP_TIMEOUT).set_preset_only());

    args.push_back(common_arg(
        {"dedup-cache-models"}, "0|1",
        "in server router mode, hide a cached model from the model list when this preset resolves to the same model file",
        [](common_params &, const std::string &) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_DEDUP_CACHE_MODELS).set_preset_only());

    // args.push_back(common_arg(
    //     {"pin"},
    //     "in server router mode, do not unload this model if models_max is exceeded",
    //     [](common_params &) { /* unused */ }
    // ).set_preset_only());
}

// file: common/arg.h
#pragma once

#include "common.h"
#include "download.h"

#include <set>
#include <map>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

// pseudo-env variable to identify preset-only arguments
#define COMMON_ARG_PRESET_LOAD_ON_STARTUP    "__PRESET_LOAD_ON_STARTUP"
#define COMMON_ARG_PRESET_STOP_TIMEOUT       "__PRESET_STOP_TIMEOUT"
#define COMMON_ARG_PRESET_DEDUP_CACHE_MODELS "__PRESET_DEDUP_CACHE_MODELS"

//
// CLI argument parsing
//

struct common_arg {
    std::set<enum llama_example> examples = {LLAMA_EXAMPLE_COMMON};
    std::set<enum llama_example> excludes = {};
    std::vector<const char *> args;
    std::vector<const char *> args_neg;  // for negated args like --no-xxx
    const char * value_hint   = nullptr; // help text or example for arg value
    const char * value_hint_2 = nullptr; // for second arg value
    const char * env          = nullptr;
    std::string help;
    bool is_sampling = false; // is current arg a sampling param?
    bool is_spec = false; // is current arg a speculative decoding param?
    bool is_preset_only = false; // is current arg preset-only (not treated as CLI arg)
    void (*handler_void)   (common_params & params) = nullptr;
    void (*handler_string) (common_params & params, const std::string &) = nullptr;
    void (*handler_str_str)(common_params & params, const std::string &, const std::string &) = nullptr;
    void (*handler_int)    (common_params & params, int) = nullptr;
    void (*handler_bool)   (common_params & params, bool) = nullptr;

    common_arg() = default;

    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(common_params & params, const std::string &)
    ) : args(args), value_hint(value_hint), help(help), handler_string(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(common_params & params, int)
    ) : args(args), value_hint(value_hint), help(help), handler_int(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const std::string & help,
        void (*handler)(common_params & params)
    ) : args(args), help(help), handler_void(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const std::initializer_list<const char *> & args_neg,
        const std::string & help,
        void (*handler)(common_params & params, bool)
    ) : args(args), args_neg(args_neg), help(help), handler_bool(handler) {}

    // support 2 values for arg
    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const char * value_hint_2,
        const std::string & help,
        void (*handler)(common_params & params, const std::string &, const std::string &)
    ) : args(args), value_hint(value_hint), value_hint_2(value_hint_2), help(help), handler_str_str(handler) {}

    common_arg & set_examples(std::initializer_list<enum llama_example> examples);
    common_arg & set_excludes(std::initializer_list<enum llama_example> excludes);
    common_arg & set_env(const char * env);
    common_arg & set_sampling();
    common_arg & set_spec();
    common_arg & set_preset_only();
    bool in_example(enum llama_example ex);
    bool is_exclude(enum llama_example ex);
    bool get_value_from_env(std::string & output) const;
    bool has_value_from_env() const;
    std::string to_string() const;

    // for using as key in std::map
    bool operator<(const common_arg& other) const {
        if (args.empty() || other.args.empty()) {
            return false;
        }
        return strcmp(args[0], other.args[0]) < 0;
    }
    bool operator==(const common_arg& other) const {
        if (args.empty() || other.args.empty()) {
            return false;
        }
        return strcmp(args[0], other.args[0]) == 0;
    }

    // get all args and env vars (including negated args/env)
    std::vector<std::string> get_args() const;
    std::vector<std::string> get_env() const;
};

namespace common_arg_utils {
    bool is_truthy(const std::string & value);
    bool is_falsey(const std::string & value);
    bool is_autoy(const std::string & value);
}

struct common_params_context {
    enum llama_example ex = LLAMA_EXAMPLE_COMMON;
    common_params & params;
    std::vector<common_arg> options;
    void(*print_usage)(int, char **) = nullptr;
    common_params_context(common_params & params) : params(params) {}
};

// parse input arguments from CLI
// if one argument has invalid value, it will automatically display usage of the specific argument (and not the full usage message)
// TODO: this function can load ggml backend (by calling llama_support_rpc)
//       this is a side-effect that should be avoided
bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);

// load all backends and print the list of available (non-CPU) devices to stdout
void common_print_available_devices();

// parse input arguments from CLI into a map
bool common_params_to_map(int argc, char ** argv, llama_example ex, std::map<common_arg, std::string> & out_map);

// populate preset-only arguments
// these arguments are not treated as command line arguments
// see: https://github.com/ggml-org/llama.cpp/issues/18163
void common_params_add_preset_options(std::vector<common_arg> & args);

struct common_models_handler {
    common_download_hf_plan plan;
    common_download_hf_plan plan_spec;
    common_download_opts opts;
};

// initialize downloading opts and hf_plan if needed, but does not download anything yet
common_models_handler common_models_handler_init(const common_params & params, llama_example curr_ex);

// check if the model is a preset repo (i.e. has a preset file)
bool common_models_handler_is_preset_repo(const common_models_handler & handler);

// download and update params with the downloaded model path
void common_models_handler_apply(common_models_handler & handler, common_params & params, common_download_callback * callback = nullptr);

// initialize argument parser context - used by test-arg-parser and preset
common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);

// file: common/build-info.h
#pragma once

#include <cstdio>

int llama_build_number(void);

const char * llama_commit(void);
const char * llama_compiler(void);

const char * llama_build_target(void);
const char * llama_build_info(void);

void llama_print_build_info(const char *, FILE * = stderr);

// file: common/chat-auto-parser-generator.cpp
#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "parsers/parsers.h"
#include "peg-parser.h"

#include <stdexcept>
#include <string>

using json = common_json;

namespace autoparser {

parser_build_context::parser_build_context(common_chat_peg_builder & p, const generation_params & inputs) :
    p(p),
    inputs(inputs),
    reasoning_parser(p.eps()) {}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs) {
    // Run differential analysis to extract template structure
    struct autoparser autoparser;
    autoparser.analyze_template(tmpl);
    return generate_parser(tmpl, inputs, autoparser);
}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs,
                                                  const autoparser &              autoparser) {
    // Create the result structure
    common_chat_params data;
    data.prompt            = common_chat_template_direct_apply(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens  = autoparser.preserved_tokens;
    data.additional_stops.insert(data.additional_stops.end(),
        autoparser.additional_stops.begin(), autoparser.additional_stops.end());

    std::string parser_generation_prompt = data.generation_prompt;

    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !inputs.continue_msg.empty()) {
        // Build up generation prompt manually
        const auto & msg = inputs.continue_msg;

        if (!autoparser.reasoning.start.empty()) {
            data.generation_prompt = data.generation_prompt.substr(0, data.generation_prompt.find(autoparser.reasoning.start));
            data.generation_prompt += autoparser.reasoning.start + msg.reasoning_content;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += autoparser.reasoning.end;
            }
        }

        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = autoparser.build_parser(inputs, parser_generation_prompt);
    data.parser = parser.save();

    // Build grammar if tools are present
    bool has_tools =
        autoparser.tools.format.mode != tool_format::NONE && inputs.tools.is_array() && !inputs.tools.empty();
    std::string trigger_marker = !autoparser.tools.format.section_start.empty() ? autoparser.tools.format.section_start :
                                                                                  autoparser.tools.format.per_call_start;

    bool has_response_format = !inputs.json_schema.empty() && inputs.json_schema.is_object();
    bool include_grammar = has_response_format || (has_tools &&
            ((inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO && !trigger_marker.empty()) ||
              inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Set grammar triggers based on tool section markers (fall back to per-call markers)
        if (data.grammar_lazy) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_marker }
            };
            if (autoparser.tools.format.openai_wrapper_trigger) {
                // model emits the OpenAI function wrapper, trigger on it
                data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "{\"type\": \"function\"," });
            }
        }
    }

    return data;
}

common_peg_arena autoparser::build_parser(const generation_params & inputs, const std::string & generation_prompt) const {
    if (!analysis_complete) {
        throw std::invalid_argument("Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)");
    }
    return build_chat_peg_parser([&](common_chat_peg_builder & p) {
        parser_build_context ctx(p, inputs);
        bool                 extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

        ctx.extracting_reasoning = extract_reasoning && reasoning.mode != reasoning_mode::NONE;
        ctx.content              = &content;
        ctx.reasoning            = &reasoning;

        // Build reasoning parser
        ctx.reasoning_parser = reasoning.build_parser(ctx);

        auto parser = p.eps();

        bool has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
        bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
        bool pure_content        = reasoning.mode == reasoning_mode::NONE;

        if (has_response_format) {
            auto response_format = p.rule("response-format", p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));
            parser = ctx.reasoning_parser + p.space() + p.choice({
                p.literal("```json") + p.space() + response_format + p.space() + p.literal("```"),
                p.space() + response_format  + p.space()
            }) + p.end();
            pure_content = false;
        } else if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && jinja_caps.supports_tool_calls) {
            parser = tools.build_parser(ctx);
            pure_content = false;
        } else {
            parser = content.build_parser(ctx);
        }
        const std::string reasoning_start = trim_whitespace(reasoning.start);
        return pure_content ? p.prefix(generation_prompt, reasoning_start) + parser : p.prefix(generation_prompt, reasoning_start) << parser;
    });
}

common_peg_parser analyze_reasoning::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (!ctx.extracting_reasoning) {
        return p.eps();
    }

    if (mode == reasoning_mode::TAG_BASED || mode == reasoning_mode::TOOLS_ONLY) {
        if (!end.empty()) {
            if (!start.empty()) {
                // Standard tag-based: optional(<think>reasoning</think>)
                return p.optional(p.optspace(start) + p.reasoning(p.until(trim_whitespace(end))) + p.optspace(end));
            }
            // Delimiter-style (empty start)
            return p.optional(p.reasoning(p.until(trim_whitespace(end))) + p.optspace(end));
        }
    }

    return p.eps();
}

common_peg_parser analyze_content::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        if (ctx.extracting_reasoning) {
            return ctx.reasoning_parser + start + p.content(p.until(end)) + end + p.end();
        }
        return p.content(p.until(start)) + start + p.content(p.until(end)) + end + p.end();
    }
    return ctx.reasoning_parser + p.content(p.rest()) + p.end();
}

common_peg_parser analyze_content::build_optional_wrapped(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        return p.optional(start + p.content(p.until(end)) + end);
    }
    return p.eps();
}

common_peg_parser analyze_tools::build_parser(parser_build_context & ctx) const {
    switch (format.mode) {
        case tool_format::JSON_NATIVE:
            return build_tool_parser_json_native(ctx);
        case tool_format::TAG_WITH_JSON:
            return build_tool_parser_tag_json(ctx);
        case tool_format::TAG_WITH_TAGGED:
            return build_tool_parser_tag_tagged(ctx);
        default:
            LOG_ERR("[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work properly. "
                "Check for a fixed template for your model in the models/templates directory of your llama.cpp installation or "
                "report an issue at https://github.com/ggml-org/llama.cpp/issues\n");
            return ctx.p.eps();
    }
}

common_peg_parser analyze_tools::build_tool_parser_json_native(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    // Build effective field names with dot notation if function_field is set
    std::string name_field = format.name_field;
    std::string args_field = format.args_field;

    if (!format.function_field.empty() && format.function_field != "function" &&
        name_field.find('.') == std::string::npos) {
        name_field = format.function_field + "." + name_field;
        args_field = format.function_field + "." + args_field;
    }

    auto tools_parser = p.eps();
    if (format.section_start.empty() && !format.per_call_start.empty()) {
        auto single_tool_parser = p.standard_json_tools(
            format.per_call_start, format.per_call_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order, format.openai_wrapper_trigger);
        tools_parser = p.trigger_rule("tool-calls", p.one_or_more(single_tool_parser + p.space()));
    } else {
        tools_parser = p.standard_json_tools(
            format.section_start, format.section_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order, format.openai_wrapper_trigger);
    }

    // Handle content wrappers if present
    if (ctx.content && ctx.content->is_always_wrapped()) {
        auto wrapped_content = ctx.content->build_optional_wrapped(ctx);
        return ctx.reasoning_parser + wrapped_content + tools_parser + p.end();
    }

    std::string tool_start = "{";
    if (!format.section_start.empty()) {
        tool_start = format.section_start;
    } else if (!format.per_call_start.empty()) {
        tool_start = format.per_call_start;
    }

    return ctx.reasoning_parser + p.optional(p.content(p.until(tool_start))) + tools_parser + p.end();
}

common_peg_parser analyze_tools::build_func_parser(common_chat_peg_builder & p, const std::string & name,
                                                    const common_peg_parser & call_id_section, bool have_call_id,
                                                    const common_peg_parser & args,
                                                    std::optional<common_peg_parser> atomic_peek) const {
    auto              open           = p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix);
    bool              matched_atomic = false;
    common_peg_parser func_parser    = p.eps();

    if (!function.args_separator.empty()) {
        open = open + p.space() + p.literal(function.args_separator);
    }

    if (!function.name_suffix.empty()) {
        func_parser    = open + call_id_section + p.space() + args;
        matched_atomic = true;
    } else if (have_call_id) {
        func_parser    = p.atomic(open + call_id_section) + p.space() + args;
        matched_atomic = true;
    } else if (atomic_peek.has_value()) {
        func_parser    = p.atomic(open + call_id_section + p.space() + *atomic_peek) + args;
        matched_atomic = true;
    } else {
        func_parser = open + call_id_section + p.space() + args;
    }

    if (!function.close.empty()) {
        func_parser = func_parser + p.space() + p.tool_close(p.literal(function.close));
    } else if (!format.per_call_end.empty()) {
        // When there's no func_close but there is a per_call_end marker, use peek() to ensure
        // we only emit tool_close when we can actually see the closing marker. This prevents
        // premature closing during partial parsing when we've seen e.g. "</" which could be
        // either "</tool_call>" (end) or "<arg_key>" prefix that failed to match.
        // Laguna (v4): the model may emit whitespace between the last </arg_value> and
        // </tool_call> even though the template renders them tight. Tolerate optional
        // leading space in the close lookahead so the tool call still closes.
        auto close_peek = arguments.tolerate_intertag_whitespace
                              ? p.peek(p.space() + p.literal(format.per_call_end))
                              : p.peek(p.literal(format.per_call_end));
        func_parser = func_parser + p.tool_close(close_peek);
    } else {
        func_parser = func_parser + p.tool_close(p.space());  // force this to process tool closing callbacks in mapper
    }
    if (!matched_atomic) {
        func_parser = p.atomic(func_parser);
    }
    return func_parser;
}

common_peg_parser analyze_tools::build_tool_parser_tag_json(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func   = tool.at("function");
        std::string  name   = func.at("name");
        const auto   schema = common_chat_tool_parameters(func);

        // Build call_id parser based on position (if supported)
        bool have_call_id = false;
        common_peg_parser call_id_section = p.eps();
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix))) + call_id.suffix;
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
            have_call_id = true;
        }
        auto args_parser = p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema));
        if (!arguments.start.empty()) {
            args_parser = p.literal(arguments.start) + args_parser;
        }
        if (!arguments.end.empty()) {
            args_parser = args_parser + p.literal(arguments.end);
        }

        auto atomic_peek = !arguments.start.empty() ? std::optional(p.peek(p.literal(arguments.start))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_parser, atomic_peek);
        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + tool_choice + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call));
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call);
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end)));
        }
    } else {
        std::string separator = ", ";  // Default
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + format.section_end);
        } else {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice + format.section_end);
        }
    }

    if (!require_calls) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
}

common_peg_parser analyze_tools::build_tool_parser_tag_tagged(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    auto until_suffix = p.rule("until-suffix", p.until(arguments.value_suffix));

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func = tool.at("function");
        std::string  name = func.at("name");

        // Build parser for each argument, separating required and optional
        std::vector<common_peg_parser> required_parsers;
        std::vector<common_peg_parser> optional_parsers;
        foreach_parameter(func, [&](const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
            auto arg =
                p.tool_arg(p.tool_arg_open(arguments.name_prefix + p.tool_arg_name(p.literal(param.name)) +
                                           arguments.name_suffix) +
                           arguments.value_prefix +
                           (param.schema->may_be_string() ?
                                p.ac(p.tool_arg_string_value(until_suffix) +
                                    p.tool_arg_close(p.literal(arguments.value_suffix)), arguments.value_suffix) :
                                (p.tool_arg_json_value(p.schema(
                                    p.json(), "tool-" + name + "-arg-" + param.name + "-schema", doc, *param.schema)) +
                                    p.tool_arg_close(p.literal(arguments.value_suffix)))));

            auto named_arg = p.rule("tool-" + name + "-arg-" + param.name, arg);
            if (param.required) {
                required_parsers.push_back(named_arg);
            } else {
                optional_parsers.push_back(named_arg);
            }
        });

        // Build required arg sequence in definition order
        common_peg_parser args_seq = p.eps();
        for (size_t i = 0; i < required_parsers.size(); i++) {
            if (i > 0) {
                args_seq = args_seq + p.space();
            }
            args_seq = args_seq + required_parsers[i];
        }

        // Build optional args with flexible ordering
        if (!optional_parsers.empty()) {
            common_peg_parser any_opt = p.choice();
            for (const auto & opt : optional_parsers) {
                any_opt |= opt;
            }
            args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
        }

        if (!arguments.start.empty()) {
            args_seq = p.literal(arguments.start) + args_seq;
        }
        if (!arguments.end.empty()) {
            args_seq = args_seq + p.literal(arguments.end);
        }

        // Build call_id parser based on position (if supported)
        common_peg_parser call_id_section = p.eps();
        bool have_call_id = false;
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            have_call_id = true;
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix)) + call_id.suffix);
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
        }

        // Only peek for an arg tag when there are required args that must follow.
        // When all args are optional, the model may emit no arg tags at all (#20650).
        auto atomic_peek = (!arguments.name_prefix.empty() && !required_parsers.empty()) ?
            std::optional(p.peek(p.literal(arguments.name_prefix))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_seq, atomic_peek);
        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + p.space() + tool_choice + p.space() + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call) + p.space());
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.space());
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end) + p.space()));
        }
    } else {
        std::string separator = ", ";  // Default

        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + p.space() + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + p.space() +
                                                         format.section_end);
        } else {
            tool_calls = p.trigger_rule(
                "tool-call", format.section_start + p.space() + tool_choice + p.space() + format.section_end);
        }
    }

    if (!require_tools) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
}

}  // namespace autoparser

// file: common/chat-auto-parser-helpers.cpp
#include "chat-auto-parser-helpers.h"

#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "log.h"
#include "peg-parser.h"

#include <cctype>
#include <numeric>

std::string trim_whitespace(const std::string & str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }

    if (start == str.length()) {
        return "";
    }

    size_t end = str.length() - 1;
    while (end > start && std::isspace(static_cast<unsigned char>(str[end]))) {
        end--;
    }

    return str.substr(start, end - start + 1);
}

std::string trim_leading_whitespace(const std::string & str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }

    return str.substr(start);
}

std::string trim_trailing_whitespace(const std::string & str) {
    if (str.empty()) {
        return "";
    }

    size_t end = str.length() - 1;
    while (end > 0 && std::isspace(static_cast<unsigned char>(str[end]))) {
        end--;
    }

    // If first char is also whitespace, return empty string
    if (end == 0 && std::isspace(static_cast<unsigned char>(str[0]))) {
        return "";
    }

    return str.substr(0, end + 1);
}

std::string trim_trailing_newlines(const std::string & str) {
    size_t end = str.length();
    while (end > 0 && str[end - 1] == '\n') {
        end--;
    }

    return str.substr(0, end);
}

static size_t common_prefix_len(const std::string & left, const std::string & right) {
    size_t prefix_len = 0;
    size_t min_len    = std::min(left.length(), right.length());
    while (prefix_len < min_len && left[prefix_len] == right[prefix_len]) {
        prefix_len++;
    }
    return prefix_len;
}

static size_t common_suffix_len(const std::string & left, const std::string & right) {
    size_t suffix_len = 0;
    size_t min_len    = std::min(left.length(), right.length());
    while (suffix_len < min_len && left[left.length() - 1 - suffix_len] == right[right.length() - 1 - suffix_len]) {
        suffix_len++;
    }
    return suffix_len;
}

diff_split calculate_diff_split(const std::string & left, const std::string & right) {
    diff_split result;

    auto left_seg = segmentize_markers(left);
    auto right_seg = segmentize_markers(right);

    if (left_seg.empty()) {
        result.right = right;
        return result;
    }
    if (right_seg.empty()) {
        result.left = left;
        return result;
    }

    auto left_start = left_seg.begin();
    auto left_end = --left_seg.end();
    auto right_start = right_seg.begin();
    auto right_end = --right_seg.end();

    auto test = [&] () {
        return left_start != left_end && right_start != right_end;
    };

    bool left_fully_consumed = false;
    bool right_fully_consumed = false;

    while (test()) {
        bool advanced = false;
        if (*left_start == *right_start) {
            result.prefix.append(left_start->value);
            left_start++;
            right_start++;
            advanced = true;
        }
        if (*left_end == *right_end) {
            result.suffix = left_end->value + result.suffix;
            if (left_start != left_end) {
                left_end--;
            } else {
                left_fully_consumed = true;
            }
            if (right_start != right_end) {
                right_end--;
            } else {
                right_fully_consumed = true;
            }
            advanced = true;
        }
        if (!advanced) {
            break;
        }
    }

    if (left_start == left_end && right_start != right_end) {
        if (*left_start == *right_end) {
            result.suffix = right_end->value + result.suffix;
            right_end--;
            left_fully_consumed = true;
        } else if (*left_start == *right_start) {
            result.prefix.append(right_start->value);
            right_start++;
            left_fully_consumed = true;
        }
    } else if (right_start == right_end && left_start != left_end) {
        if (*left_end == *right_start) {
            result.suffix = left_end->value + result.suffix;
            left_end--;
            right_fully_consumed = true;
        } else if (*left_start == *right_start) {
            result.prefix.append(left_start->value);
            left_start++;
            right_fully_consumed = true;
        }
    } else if (left_start == left_end && right_start == right_end && *left_start == *right_start && left_start->type == segment_type::MARKER) {
        result.prefix.append(right_start->value);
        left_fully_consumed = true;
        right_fully_consumed = true;
    }

    auto eat_segment = [](std::string str, const segment & seg) -> std::string { return std::move(str) + seg.value; };

    bool can_have_text_suffix = left_end->type == segment_type::TEXT && right_end->type == segment_type::TEXT;
    bool can_have_text_prefix = right_start->type == segment_type::TEXT && left_start->type == segment_type::TEXT;

    std::string remainder_left = std::accumulate(left_start, left_fully_consumed ? left_end : ++left_end, std::string(), eat_segment);
    std::string remainder_right = std::accumulate(right_start, right_fully_consumed ? right_end : ++right_end, std::string(), eat_segment);

    size_t suffix_len = can_have_text_suffix ? common_suffix_len(remainder_left, remainder_right) : 0;
    // avoid overlaps between prefix and suffix
    size_t prefix_len = can_have_text_prefix ? common_prefix_len(remainder_left.substr(0, remainder_left.size() - suffix_len),
        remainder_right.substr(0, remainder_right.size() - suffix_len)) : 0;

    result.prefix.append(remainder_left.substr(0, prefix_len));
    result.suffix = remainder_left.substr(remainder_left.length() - suffix_len, suffix_len) + result.suffix;
    result.left = remainder_left.substr(prefix_len, remainder_left.length() - prefix_len - suffix_len);
    result.right = remainder_right.substr(prefix_len, remainder_right.length() - prefix_len - suffix_len);

    if (result.left == "" && result.right == "") {
        // degenerate case, no diff
        result.prefix = left;
        result.suffix = "";
        // pick prefix = all as representation
    }

    // When left has no unique content (result.left is empty), left is entirely
    // shared with right. The simultaneous prefix/suffix segment matching can
    // incorrectly consume trailing segments of left as suffix when those same
    // segments also appear at the end of right (e.g. "\n" at the end of both
    // the shared content and the generation prompt). This rotates the diff.
    // Fix: if left is a prefix of right, enforce that directly.
    if (result.left.empty() && !result.right.empty() &&
            left.size() <= right.size() &&
            right.substr(0, left.size()) == left) {
        result.prefix = left;
        result.suffix = "";
        result.right  = right.substr(left.size());
    }

    return result;
}

// Returns the prefix of `full` up until the first occurrence of the common prefix of `left` and `right`
std::string until_common_prefix(const std::string & full, const std::string & left, const std::string & right) {
    // Find the common prefix of left and right
    size_t common_prefix_len = 0;
    size_t min_len           = std::min(left.length(), right.length());
    while (common_prefix_len < min_len && left[common_prefix_len] == right[common_prefix_len]) {
        common_prefix_len++;
    }

    // If there's no common prefix, return empty string
    if (common_prefix_len == 0) {
        return "";
    }

    // Find the common prefix in the full string
    std::string common_prefix = left.substr(0, common_prefix_len);
    size_t      pos           = full.find(common_prefix);

    // If not found, return empty string
    if (pos == std::string::npos) {
        return "";
    }

    // Return everything before the common prefix
    return full.substr(0, pos);
}

// Returns the suffix of `full` after the last occurrence of the common suffix of `left` and `right`
std::string after_common_suffix(const std::string & full, const std::string & left, const std::string & right) {
    // Find the common suffix of left and right (compare from the end)
    size_t common_suffix_len = 0;
    size_t min_len           = std::min(left.length(), right.length());
    while (common_suffix_len < min_len &&
           left[left.length() - 1 - common_suffix_len] == right[right.length() - 1 - common_suffix_len]) {
        common_suffix_len++;
    }

    // If there's no common suffix, return empty string
    if (common_suffix_len == 0) {
        return "";
    }

    // Extract the common suffix
    std::string common_suffix = left.substr(left.length() - common_suffix_len);

    // Find the last occurrence of the common suffix in the full string
    size_t pos = full.rfind(common_suffix);

    // If not found, return empty string
    if (pos == std::string::npos) {
        return "";
    }

    // Return everything after the common suffix
    return full.substr(pos + common_suffix_len);
}

// TODO: segmentize will treat a JSON array inside tags as a tag: <calls>[{ "fun": { ... } }]</calls> will be three markers
// not too worried about that because it hasn't turned out as a problem anywhere, but noting here in case it will
// Might have to put some restrictions on tag contents as well (like "no { }")
std::vector<segment> segmentize_markers(const std::string & text) {
    std::vector<segment> retval;
    bool in_marker = false;
    char marker_opener = '\0';

    auto is_marker_opener = [](char c) -> bool { return c == '<' || c == '['; };
    auto is_marker_closer = [](char op, char c) -> bool { return (op == '<' && c == '>') || (op == '[' && c == ']'); };

    size_t last_border = 0;

    for (size_t cur_pos = 0; cur_pos < text.length(); cur_pos++) {
        if (!in_marker && is_marker_opener(text[cur_pos])) {
            if (last_border < cur_pos) {
                retval.push_back(segment(segment_type::TEXT, text.substr(last_border, cur_pos - last_border)));
            }
            last_border = cur_pos;
            in_marker = true;
            marker_opener = text[cur_pos];
        } else if (in_marker && is_marker_closer(marker_opener, text[cur_pos])) {
            // no need to check because last_border will always be smaller
                retval.push_back(segment(segment_type::MARKER, text.substr(last_border, cur_pos - last_border + 1)));
            last_border = cur_pos + 1;
            in_marker = false;
            marker_opener = '\0';
        }
    }
    if (last_border < text.length()) {
            retval.push_back(segment(segment_type::TEXT, text.substr(last_border)));
    }
    return retval;
}

std::vector<segment> prune_whitespace_segments(const std::vector<segment> & segments) {
    std::vector<segment> result;
    for (const auto & seg : segments) {
        if (!trim_whitespace(seg.value).empty()) {
            result.push_back(seg);
        }
    }
    return result;
}

namespace autoparser {

static const std::string ERR_TMPL = "#**ERROR**#";

std::string apply_template(const common_chat_template & tmpl, const template_params & params) {
    generation_params tmpl_params;
    tmpl_params.messages              = params.messages;
    tmpl_params.tools                 = params.tools;
    tmpl_params.add_generation_prompt = params.add_generation_prompt;
    tmpl_params.enable_thinking       = params.enable_thinking;

    if (params.extra_context) {
        tmpl_params.extra_context = *params.extra_context;
    }
    tmpl_params.extra_context["enable_thinking"] = params.enable_thinking;

    try {
        return common_chat_template_direct_apply(tmpl, tmpl_params);
    } catch (const std::exception & e) {
        LOG_DBG("Template application failed: %s\n", e.what());
        return ERR_TMPL;
    }
}

std::optional<compare_variants_result> compare_variants(
    const common_chat_template &                   tmpl,
    const template_params &                        params_A,
    const std::function<void(template_params &)> & params_modifier) {
    // Create variant B by copying A
    template_params params_B = params_A;

    // Apply modifier to create variant B
    if (params_modifier) {
        params_modifier(params_B);
    }

    // Apply template to both variants
    std::string output_A = apply_template(tmpl, params_A);
    std::string output_B = apply_template(tmpl, params_B);

    // Check for template application failures
    if (output_A == ERR_TMPL || output_B == ERR_TMPL) {
        return std::nullopt;
    }

    // Calculate diff and return result with both outputs
    compare_variants_result result;
    result.diff     = calculate_diff_split(output_A, output_B);
    result.output_A = output_A;
    result.output_B = output_B;

    return result;
}

}  // namespace autoparser


// file: common/chat-auto-parser-helpers.h
#pragma once

#include "chat-auto-parser.h"

#include <functional>
#include <optional>
#include <string>

std::string trim_whitespace(const std::string & str);
std::string trim_leading_whitespace(const std::string & str);
std::string trim_trailing_whitespace(const std::string & str);
std::string trim_trailing_newlines(const std::string & str);

// calculate a diff split (longest common prefix, longest common suffix excluding prefix,
// mismatched part on the left, mismatched part on the right) between two strings
// account for markers - align prefix and suffix endings so that they end on markers
// * eg.:
// calculate_diff_split("<html><body><div></div></body></html>", "<html><body><p>Something</p></body><html>") ->
//  { "prefix": "<html><body>" (not: "<html><body><"), "suffix": "</body></html>", "left": "<div></div>", "right": "<p>Something</p>" }
// calculate_diff_split("<html><body>Something</body></html>", "<html><body></body><html>") ->
//  { "prefix": "<html><body>", "suffix": "</body></html>", "left": "Something", "right": "" }
diff_split calculate_diff_split(const std::string & left, const std::string & right);

// Returns the prefix of `full` up until the first occurrence of the common prefix of `left` and `right`
// Returns empty string if there's no common prefix
// * eg.:
// until_common_prefix("really want a FUNCTION call", "FUNCTION alpha", "FUNCTION beta") -> "really want a "
// until_common_prefix("<tool_call>", "<something>", "<something_else>") -> ""
// until_common_prefix("some text", "1234", "abcd") -> ""
// until_common_prefix("one arg two args three args four", "argument alpha", "argument beta") -> "one ""
std::string until_common_prefix(const std::string & full, const std::string & left, const std::string & right);

// Returns the suffix of `full` after the last occurrence of the common suffix of `left` and `right`
// Returns empty string if there's no common suffix
// Mirror function of `until_common_prefix`
// * eg.:
// after_common_suffix("really want a FUNCTION call", "first FUNCTION", "second FUNCTION") -> " call"
// after_common_suffix("one arg two-args three args four", "alpha-args", "beta-args") -> " three args four"
std::string after_common_suffix(const std::string & full, const std::string & left, const std::string & right);

// Segmentize text into markers and non-marker fragments
// * eg.:
// segmentize_markers("<html><head><title>The site title</title><body><div>Here's some <b>content</b></div></body></html>" ->
//  [ (MARKER, "<html>"), (MARKER, "<head>"), (MARKER, "<title>"), (TEXT, "The site title"), (MARKER, "</title>"),
//    (MARKER, "<body>"), (MARKER, "<div>"), (TEXT, "Here's some "), (MARKER, "<b>"), (TEXT, "content"), (MARKER, "</b>"),
//    (MARKER, "</div>"), (MARKER, "</body>"), (MARKER, "</html>")
//  ]
// segmentize_markers("<|tool_call|>[args]{ are here }[/args]<|tool_call_end|>") ->
//  [ (MARKER, "<|tool_call|>"), (MARKER, "[args]"), (TEXT, "{ are here }"), (MARKER, "[/args]"), (MARKER, "<|tool_call_end|>") ]
std::vector<segment> segmentize_markers(const std::string & text);

// Prune whitespace-only segments from a vector of segments
// * eg.:
// segmentize_markers("<tool_call>\n<function=foo>\n<arg=bar>\n   \n</arg>\n</function>\n</tool_call>") ->
//  X = [ (MARKER, "<tool_call>"), (TEXT, "\n"), (MARKER, "<function=foo>"), (TEXT, "\n"), (MARKER, "<arg=bar>"), (TEXT, "\n   \n"),
//        (MARKER, "</arg>"), (TEXT, "\n"), (MARKER, "</function>"), (TEXT, "\n"), (MARKER, "</tool_call>") ]
// prune_whitespace_segments(X) -> [ (MARKER, "<tool_call>"), (MARKER, "<function=foo>"), (MARKER, "<arg=bar>"), (MARKER, "</arg>"),
//                                   (MARKER, "</function>"), (MARKER, "</tool_call>") ]
std::vector<segment> prune_whitespace_segments(const std::vector<segment> & segments);

namespace autoparser {

// Apply a template with the given parameters, returning the rendered string (empty on failure)
std::string apply_template(const common_chat_template & tmpl, const template_params & params);

// Factorized differential comparison function
// Takes base params and a single modifier lambda to create variant B
// Returns compare_variants_result containing diff and both outputs, or std::nullopt on failure
std::optional<compare_variants_result> compare_variants(
    const common_chat_template &                   tmpl,
    const template_params &                        params_A,
    const std::function<void(template_params &)> & params_modifier);

}  // namespace autoparser

// file: common/chat-auto-parser.h
#pragma once

#include "chat.h"
#include "common.h"
#include "jinja/caps.h"
#include "peg-parser.h"
#include "json.h"

#include <chrono>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using json = common_json;

class common_chat_peg_builder;

// ============================================================================
// Parameters for template application (low-level, used by diff analysis)
// ============================================================================
struct template_params {
    json                messages;
    json                tools;
    bool                add_generation_prompt = false;
    bool                enable_thinking       = true;
    std::optional<json> extra_context         = std::nullopt;
};

struct diff_split {
    std::string prefix;
    std::string suffix;
    std::string left;
    std::string right;

    bool operator==(struct diff_split & other) const {
        return prefix == other.prefix && suffix == other.suffix && left == other.left && right == other.right;
    }
};

// Result of compare_variants containing diff and original outputs
struct compare_variants_result {
    diff_split  diff;
    std::string output_A;
    std::string output_B;
};

namespace autoparser {

// ============================================================================
// High-level params for parser generation
// ============================================================================

struct generation_params {
    json                                  messages;
    json                                  tools;
    common_chat_tool_choice               tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    json                                  json_schema;
    bool                                  parallel_tool_calls = true;
    common_reasoning_format               reasoning_format    = COMMON_REASONING_FORMAT_AUTO;
    bool                                  stream              = true;
    std::string                           grammar;
    bool                                  add_generation_prompt  = false;
    common_chat_continuation              continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    common_chat_msg                       continue_msg;
    bool                                  enable_thinking        = true;
    std::chrono::system_clock::time_point now                    = std::chrono::system_clock::now();
    json                                  extra_context;
    bool                                  add_bos       = false;
    bool                                  add_eos       = false;
    bool                                  is_inference  = true;
    bool                                  add_inference = false;
    bool                                  mark_input    = true;  // whether to mark input strings in the jinja context

    bool has_continuation() const {
        return continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !continue_msg.empty();
    }
};

// ============================================================================
// Analysis Result Enums
// ============================================================================

// Reasoning handling mode (derived from R1-R3 comparisons)
enum class reasoning_mode {
    NONE,           // No reasoning markers detected
    TAG_BASED,      // Tag-based: <think>...</think> (start can be empty for delimiter-style)
    TOOLS_ONLY      // Only reason on tool calls, not on normal content
};

inline std::ostream & operator<<(std::ostream & os, const reasoning_mode & mode) {
    switch (mode) {
        case reasoning_mode::NONE:
            return os << "NONE";
        case reasoning_mode::TAG_BASED:
            return os << "TAG_BASED";
        case reasoning_mode::TOOLS_ONLY:
            return os << "TOOLS_ONLY";
        default:
            return os << "UNKNOWN";
    }
}

// Content wrapping mode (derived from C1 comparison)
enum class content_mode {
    PLAIN,                   // No content markers
    ALWAYS_WRAPPED,          // Content always wrapped with markers
    WRAPPED_WITH_REASONING,  // Content wrapped only when reasoning present
};

inline std::ostream & operator<<(std::ostream & os, const content_mode & mode) {
    switch (mode) {
        case content_mode::PLAIN:
            return os << "PLAIN";
        case content_mode::ALWAYS_WRAPPED:
            return os << "ALWAYS_WRAPPED";
        case content_mode::WRAPPED_WITH_REASONING:
            return os << "WRAPPED_WITH_REASONING";
        default:
            return os << "UNKNOWN";
    }
}

// Call ID position in tool calls (for non-JSON formats)
enum class call_id_position {
    NONE,                   // No call ID support detected
    PRE_FUNC_NAME,          // Call ID before function name: [CALL_ID]id[FUNC]name{args}
    BETWEEN_FUNC_AND_ARGS,  // Call ID between function and args: [FUNC]name[CALL_ID]id{args}
    POST_ARGS,              // Call ID after arguments: [FUNC]name{args}[CALL_ID]id
};

inline std::ostream & operator<<(std::ostream & os, const call_id_position & pos) {
    switch (pos) {
        case call_id_position::NONE:
            return os << "NONE";
        case call_id_position::PRE_FUNC_NAME:
            return os << "PRE_FUNC_NAME";
        case call_id_position::BETWEEN_FUNC_AND_ARGS:
            return os << "BETWEEN_FUNC_AND_ARGS";
        case call_id_position::POST_ARGS:
            return os << "POST_ARGS";
        default:
            return os << "UNKNOWN";
    }
}

// Tool call format classification (derived from T1-T5, A1-A3 comparisons)
enum class tool_format {
    NONE,             // No tool support detected
    JSON_NATIVE,      // Pure JSON: {"name": "X", "arguments": {...}}
    TAG_WITH_JSON,    // Tag-based with JSON args: <function=X>{...}</function>
    TAG_WITH_TAGGED,  // Tag-based with tagged args: <param=key>value</param>
};

inline std::ostream & operator<<(std::ostream & os, const tool_format & format) {
    switch (format) {
        case tool_format::NONE:
            return os << "NONE";
        case tool_format::JSON_NATIVE:
            return os << "JSON_NATIVE";
        case tool_format::TAG_WITH_JSON:
            return os << "TAG_WITH_JSON";
        case tool_format::TAG_WITH_TAGGED:
            return os << "TAG_WITH_TAGGED";
        default:
            return os << "UNKNOWN";
    }
}

// ============================================================================
// Sub-structs for tool analysis
// ============================================================================

struct tool_format_analysis {
    tool_format mode = tool_format::NONE;

    std::string section_start;   // e.g., "<tool_call>", "[TOOL_CALLS]", ""
    std::string section_end;     // e.g., "</tool_call>", ""
    std::string per_call_start;  // e.g., "<|tool_call_begin|>", "" (for multi-call templates)
    std::string per_call_end;    // e.g., "<|tool_call_end|>", ""

    bool fun_name_is_key = false;       // In JSON format function name is JSON key, i.e. { "<funname>": { ... arguments ... } }
    bool tools_array_wrapped = false;   // Tool calls wrapped in JSON array [...]
    bool openai_wrapper_trigger = false;  // model emits the OpenAI function wrapper, trigger on it

    std::string              function_field = "function";
    std::string              name_field     = "name";
    std::string              args_field     = "arguments";
    std::string              id_field;
    std::string              gen_id_field;
    std::vector<std::string> parameter_order;
};

struct tool_function_analysis {
    std::string name_prefix;     // e.g., "<function=", "\"name\": \"", "functions."
    std::string name_suffix;     // e.g., ">", "\"", ":0"
    std::string args_separator;  // e.g., "<tool_sep>" (marker between function name and arguments)
    std::string close;           // e.g., "</function>", "" (for tag-based)
};

struct tool_arguments_analysis {
    std::string start;          // e.g., "<|tool_call_argument_begin|>", "<args>"
    std::string end;            // e.g., "<|tool_call_argument_end|>", "</args>"
    std::string name_prefix;   // e.g., "<param=", "<arg_key>", "\""
    std::string name_suffix;   // e.g., ">", "</arg_key>", "\":"
    std::string value_prefix;  // e.g., "", "<arg_value>", ""
    std::string value_suffix;  // e.g., "</param>", "</arg_value>", ""
    std::string separator;     // e.g., "", "\n", ","
    bool tolerate_intertag_whitespace = false; // Laguna: accept optional whitespace between arg tags
};

struct tool_id_analysis {
    call_id_position pos = call_id_position::NONE;

    std::string prefix;  // e.g., "[CALL_ID]" (marker before call ID value)
    std::string suffix;  // e.g., "" (marker after call ID value, before next section)
};

// ============================================================================
// Parser build context (shared interface for build_parser methods)
// ============================================================================

struct analyze_content;
struct analyze_reasoning;

struct parser_build_context {
    common_chat_peg_builder & p;
    const generation_params &         inputs;
    common_peg_parser                 reasoning_parser;
    bool                              extracting_reasoning = false;
    const analyze_reasoning *         reasoning            = nullptr;
    const analyze_content *           content              = nullptr;

    parser_build_context(common_chat_peg_builder & p, const generation_params & inputs);
};

// ============================================================================
// Base class for analyzers with parser building
// ============================================================================

struct analyze_base {
    virtual ~analyze_base() = default;
    virtual common_peg_parser build_parser(parser_build_context & ctx) const = 0;

  protected:
    const common_chat_template * tmpl = nullptr;

    analyze_base() = default;
    explicit analyze_base(const common_chat_template & tmpl) : tmpl(&tmpl) {}
};

// ============================================================================
// Reasoning analyzer
// ============================================================================

struct analyze_reasoning : analyze_base {
    reasoning_mode mode = reasoning_mode::NONE;

    std::string start;  // e.g., "<think>", "[THINK]", "<|START_THINKING|>", ""
    std::string end;    // e.g., "</think>", "[BEGIN FINAL RESPONSE]", "<|END_THINKING|>"

    analyze_reasoning() = default;
    analyze_reasoning(const common_chat_template & tmpl, bool supports_tools);
    analyze_reasoning(std::string start_, std::string end_) : start(std::move(start_)), end(std::move(end_)) {}

    common_peg_parser build_parser(parser_build_context & ctx) const override;

  private:
    // Look for reasoning markers in rendered content
    void compare_reasoning_presence();

    // Compare generation prompt with enable_thinking=true vs false
    void compare_thinking_enabled();

    // Check if reasoning is always possible or only in tool calls
    void compare_reasoning_scope();
};

// ============================================================================
// Content analyzer
// ============================================================================

struct analyze_content : analyze_base {
    content_mode mode = content_mode::PLAIN;

    std::string start;  // e.g., "<response>", ">>>all\n", ""
    std::string end;    // e.g., "</response>", ""

    bool requires_nonnull_content = false;

    analyze_content() = default;
    analyze_content(const common_chat_template & tmpl, const analyze_reasoning & reasoning);

    common_peg_parser build_parser(parser_build_context & ctx) const override;

    bool is_always_wrapped() const;
    common_peg_parser build_optional_wrapped(parser_build_context & ctx) const;
};

// ============================================================================
// Tool analyzer
// ============================================================================

struct analyze_tools : analyze_base {
    tool_format_analysis    format;
    tool_function_analysis  function;
    tool_arguments_analysis arguments;
    tool_id_analysis        call_id;

    analyze_tools() = default;
    analyze_tools(const common_chat_template & tmpl,
                  const jinja::caps &          caps,
                  const analyze_reasoning &    reasoning);

    common_peg_parser build_parser(parser_build_context & ctx) const override;

  private:
    // Extract tool calling 'haystack' for further analysis and delegate further analysis based on format
    void analyze_tool_calls(const analyze_reasoning & reasoning, bool supports_parallel_tool_calls);

    // Analyze format based on position of function and argument name in needle
    void analyze_tool_call_format(const std::string &       haystack,
                                  const std::string &       fun_name_needle,
                                  const std::string &       arg_name_needle,
                                  const analyze_reasoning & reasoning,
                                  bool                      supports_parallel_tool_calls);

    // Analyze specifics of JSON native format (entire tool call is a JSON object)
    void analyze_tool_call_format_json_native(const std::string & clean_haystack,
                                              const std::string & fun_name_needle,
                                              const std::string & arg_name_needle);

    // Check if parallel calls in JSON native format array wrapped or tag wrapped
    void analyze_json_native_parallel_calls();

    // Analyze specifics of non-JSON native format (tags for function name or for function name and arguments)
    void analyze_tool_call_format_non_json(const std::string & clean_haystack,
                                           const std::string & fun_name_needle);

    // Check for and extract specific per-call markers for non-native-JSON templates with parallel call support
    void check_per_call_markers();

    // Extract function name markers
    void extract_function_markers();

    // Delegates to separate functions for: separator analysis, argument name analysis, argument value analysis
    void analyze_arguments();

    // Extract argument name markers
    void extract_argument_name_markers();

    // Extract argument value markers
    void extract_argument_value_markers();

    // Extract argument separator, if specified (eg. <arg=foo>...</arg><sep><arg=bar>...</arg>)
    void extract_argument_separator();

    // Extract argument wrapper markers, if present (eg. '<args><arg=foo>...</arg><arg=bar>...</arg></args>')
    void extract_args_markers();

    // Extract call ID markers, if present
    void extract_call_id_markers();

    // Per-format tool parser builders
    common_peg_parser build_tool_parser_json_native(parser_build_context & ctx) const;
    common_peg_parser build_tool_parser_tag_json(parser_build_context & ctx) const;
    common_peg_parser build_tool_parser_tag_tagged(parser_build_context & ctx) const;

    // Shared helper: builds func_parser from open+call_id+args, handling atomic wrapping and close.
    // atomic_peek: if present, used as the peek expression in the third atomicity branch.
    common_peg_parser build_func_parser(common_chat_peg_builder & p, const std::string & name,
                                        const common_peg_parser & call_id_section, bool have_call_id,
                                        const common_peg_parser & args,
                                        std::optional<common_peg_parser> atomic_peek) const;
};

// ============================================================================
// Main autoparser class
// ============================================================================

struct autoparser {
    jinja::caps          jinja_caps;
    std::string          user_start;
    std::string          assistant_start;
    analyze_reasoning    reasoning;
    analyze_content      content;
    analyze_tools        tools;
    bool                 analysis_complete = false;

    // Preserved tokens for tokenizer (union of all non-empty markers)
    std::vector<std::string> preserved_tokens;
    std::vector<std::string> additional_stops;  // literal stop strings (e.g. Laguna </assistant>) caught however tokenized

    autoparser() = default;

    // Find the starting marker for the user message and assistant message
    std::string detect_user_start_marker(const common_chat_template & tmpl);
    std::string detect_assistant_start_marker(const common_chat_template & tmpl);

    // Run full differential analysis on a template
    void analyze_template(const common_chat_template & tmpl);

    // Build the PEG parser for this template
    common_peg_arena build_parser(const generation_params & inputs, const std::string & generation_prompt) const;

  private:
    // Collect tokens from entire analysis to preserve
    void collect_preserved_tokens();
};

// ============================================================================
// Parser generator
// ============================================================================

class peg_generator {
  public:
    static common_chat_params generate_parser(const common_chat_template &    tmpl,
                                              const struct generation_params & inputs);

    static common_chat_params generate_parser(const common_chat_template &    tmpl,
                                              const struct generation_params & inputs,
                                              const autoparser &              autoparser);
};

}  // namespace autoparser

enum segment_type { TEXT, MARKER };

inline std::ostream & operator<<(std::ostream & os, const segment_type & type) {
    switch (type) {
        case segment_type::TEXT:
            return os << "TEXT";
        case segment_type::MARKER:
            return os << "MARKER";
        default:
            return os << "UNKNOWN";
    }
}

struct segment {
    segment_type type;
    std::string  value;

    segment(segment_type type, std::string value) : type(type), value(std::move(value)) {}

    bool operator==(const segment & other) const {
        return type == other.type && value == other.value;
    }

    bool operator!=(const segment & other) const {
        return !(*this == other);
    }
};

// file: common/chat-diff-analyzer.cpp
#include "chat-auto-parser.h"
#include "chat-auto-parser-helpers.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "peg-parser.h"

#include <algorithm>
#include <cctype>
#include <numeric>
#include <ostream>
#include <sstream>

#define ANSI_RESET  "\033[0m"
#define ANSI_PURPLE "\033[1m\x1b[38;5;126m"
#define ANSI_ORANGE "\033[1m\x1b[38;5;214m"
#define ANSI_RED    "\033[1m\x1b[38;5;196m"

using json = common_json;

namespace autoparser {

static const std::string FUN_FIRST = "FFF_FIRST_FUN_F";
static const std::string FUN_SECOND = "SSS_SECOND_FUN_S";
static const std::string ARG_FIRST = "AA_ARG_FST_AA";
static const std::string ARG_SECOND = "BB_ARG_SND_BB";
static const std::string USER_MSG = "U_USER_MSG Hello END_U";
static const std::string USER_MSG_TWO = "V_USER_MSG Hello END_V";
static const std::string ASSISTANT_MSG = "A_ASST_MSG I can help END_A";
static const std::string THINKING_CONTENT = "REASON_PART I am thinking END_R";
static const std::string CALL_ID_001 = "call00001";
static const std::string CALL_ID_002 = "call00002";
static const std::string CALL_ID_999 = "call99999";

static std::vector<std::function<void(const common_chat_template & tmpl, autoparser &)>> workarounds(
    { // Old reasoning Qwen templates - they don't really display reasoning content, but we still want to
      // support reasoning on them
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("content.split('</think>')") != std::string::npos &&
              tmpl.src.find("reasoning_content") == std::string::npos &&
              tmpl.src.find("<SPECIAL_12>") == std::string::npos &&
              analysis.reasoning.mode == reasoning_mode::NONE) {
              analysis.reasoning.mode  = reasoning_mode::TAG_BASED;
              analysis.reasoning.start = "<think>";
              analysis.reasoning.end   = "</think>";
              analysis.preserved_tokens.push_back("<think>");
              analysis.preserved_tokens.push_back("</think>");
              LOG_DBG(ANSI_ORANGE "[Patch: old Qwen/Deepseek thinking template]\n" ANSI_RESET);
          }
      },
      // Granite 3.3, with separate reasoning and content markers
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("Write your thoughts between <think></think> and write your response between "
                            "<response></response>") != std::string::npos) {
              analysis.reasoning.mode  = reasoning_mode::TAG_BASED;
              analysis.reasoning.start = "<think>";
              analysis.reasoning.end   = "</think>";
              analysis.preserved_tokens.push_back("<think>");
              analysis.preserved_tokens.push_back("</think>");
              analysis.content.mode  = content_mode::WRAPPED_WITH_REASONING;
              analysis.content.start = "<response>";
              analysis.content.end   = "</response>";
              analysis.preserved_tokens.push_back("<response>");
              analysis.preserved_tokens.push_back("</response>");
              LOG_DBG(ANSI_ORANGE "[Patch: Granite 3.3]\n" ANSI_RESET);
          }
      },
      // Cohere Command R+ - content wrapped in <|CHATBOT_TOKEN|>...<|END_OF_TURN_TOKEN|>
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("<|CHATBOT_TOKEN|>") != std::string::npos &&
              tmpl.src.find("<|END_OF_TURN_TOKEN|>") != std::string::npos && analysis.content.start.empty()) {
              analysis.content.mode  = content_mode::ALWAYS_WRAPPED;
              analysis.content.start = "<|CHATBOT_TOKEN|>";
              analysis.content.end   = "<|END_OF_TURN_TOKEN|>";
              analysis.preserved_tokens.push_back("<|CHATBOT_TOKEN|>");
              analysis.preserved_tokens.push_back("<|END_OF_TURN_TOKEN|>");
              analysis.user_start = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>";
              LOG_DBG(ANSI_ORANGE "[Patch: Cohere Command R+]\n" ANSI_RESET);
          }
      },
      // Functionary - no tool call section delimiter
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("set has_code_interpreter = tools | selectattr(\"type\", \"equalto\", "
                            "\"code_interpreter\") | list | length > 0") != std::string::npos) {
              analysis.content.mode                = content_mode::PLAIN;
              analysis.content.end                 = "";
              analysis.tools.function.name_prefix  = "";
              analysis.tools.format.section_start  = "";
              analysis.tools.format.section_end    = "";
              analysis.tools.format.per_call_start = "<function=";
              analysis.tools.format.per_call_end   = "</function>";
              analysis.tools.function.close        = "";
              analysis.preserved_tokens.clear();
              analysis.preserved_tokens.push_back("<|eot_id|>");
              analysis.preserved_tokens.push_back("<|eom_id|>");
              analysis.preserved_tokens.push_back("<function=");
              analysis.preserved_tokens.push_back(">");
              analysis.preserved_tokens.push_back("</function>");
              LOG_DBG(ANSI_ORANGE "[Patch: Functionary 3.1]\n" ANSI_RESET);
          }
      },
      // DeepSeek-R1-Distill-Qwen
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find(
                  "{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>'") !=
              std::string::npos) {
              analysis.tools.format.section_start  = "<｜tool▁calls▁begin｜>";
              analysis.tools.format.section_end    = "<｜tool▁calls▁end｜>";
              analysis.tools.format.per_call_start = "<｜tool▁call▁begin｜>function";
              analysis.tools.function.name_prefix  = "<｜tool▁sep｜>";
              analysis.tools.format.per_call_end   = "<｜tool▁call▁end｜>";
              analysis.tools.function.close        = "```";
              LOG_DBG(ANSI_ORANGE "[Patch: DeepSeek-R1-Distill-Qwen]\n" ANSI_RESET);
          }
      },
      // Nemotron Nano v2
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("<SPECIAL_10>") != std::string::npos && tmpl.src.find("<SPECIAL_11>") != std::string::npos &&
              tmpl.src.find("<SPECIAL_12>") != std::string::npos && tmpl.src.find("<TOOL_RESPONSE>") != std::string::npos) {

              analysis.tools.format.mode           = tool_format::JSON_NATIVE;
              analysis.tools.format.section_start  = "";
              analysis.tools.format.section_end    = "";
              analysis.tools.format.per_call_start = "<TOOLCALL>";
              analysis.tools.format.per_call_end   = "</TOOLCALL>";
              analysis.tools.format.tools_array_wrapped = true;
              analysis.content.mode                = content_mode::PLAIN;
              analysis.content.start               = "";
              analysis.content.end                 = "";
              analysis.reasoning.mode              = reasoning_mode::TAG_BASED;
              analysis.reasoning.start             = "<think>\n";
              analysis.reasoning.end               = "</think>";
              analysis.assistant_start             = "<SPECIAL_11>Assistant";
              analysis.user_start                  = "<SPECIAL_11>User";
              analysis.preserved_tokens.clear();
              analysis.preserved_tokens.push_back("<SPECIAL_11>");
              analysis.preserved_tokens.push_back("</think>");
              analysis.preserved_tokens.push_back("<TOOLCALL>");
              analysis.preserved_tokens.push_back("</TOOLCALL>");
              LOG_DBG(ANSI_ORANGE "[Patch: Nemotron Nano v2]\n" ANSI_RESET);
          }
      },
      // Fireworks
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("{%- set system_prompt = '<|start_header_id|>' + 'system' + '<|end_header_id|>\\n\\n'"
            " + message['content'] | trim + '\\n' + system_prompt_suffix + '<|eot_id|>' -%}") != std::string::npos) {
              analysis.assistant_start             = "<|start_header_id|>assistant<|end_header_id|>";
              analysis.user_start                  = "<|start_header_id|>user<|end_header_id|>";
              LOG_DBG(ANSI_ORANGE "[Patch: Fireworks v2]\n" ANSI_RESET);
          }
      },
      // Solar Open
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("<|begin|>assistant<|think|><|end|>") != std::string::npos) {
              analysis.assistant_start             = "<|begin|>assistant";
              LOG_DBG(ANSI_ORANGE "[Patch: Solar Open]\n" ANSI_RESET);
          }
      },
      // Apriel 1.6
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("if not loop.last and '[BEGIN FINAL RESPONSE]' in asst_text") != std::string::npos) {
              analysis.user_start                  = "<|begin_user|>";
              analysis.assistant_start             = "<|begin_assistant|>";
              LOG_DBG(ANSI_ORANGE "[Patch: Apriel 1.6]\n" ANSI_RESET);
          }
      },
      // template uses the JSON {name, parameters} tool instruction, emits the OpenAI function wrapper
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("Respond in the format {\"name\": function name") != std::string::npos &&
              tmpl.src.find("Do not use variables.") != std::string::npos) {
              analysis.tools.format.openai_wrapper_trigger = true;
              LOG_DBG(ANSI_ORANGE "[Patch: JSON name/parameters tool instruction]\n" ANSI_RESET);
          }
      },
      // Laguna (poolside) - the v4 chat template renders reasoning and tool-arg
      // delimiters with formatting whitespace ("<think>\n", "</arg_value>\n") that
      // the model does not emit, so the inferred delimiters carry a spurious
      // newline and never match the model output. Trim to the bare tag. (v8
      // renders without the whitespace, so this is a no-op there.)
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("laguna_glm_thinking") != std::string::npos) {
              analysis.reasoning.start              = trim_whitespace(analysis.reasoning.start);
              analysis.reasoning.end                = trim_whitespace(analysis.reasoning.end);
              analysis.tools.arguments.value_prefix = trim_whitespace(analysis.tools.arguments.value_prefix);
              analysis.tools.arguments.value_suffix = trim_whitespace(analysis.tools.arguments.value_suffix);
              analysis.tools.arguments.separator    = trim_whitespace(analysis.tools.arguments.separator);
              analysis.tools.arguments.tolerate_intertag_whitespace = true;
              // The CONTROL/eot </assistant> token only halts generation when emitted as the
              // single token; after tool calls the model can spell it out as text tokens.
              // A literal stop string catches it either way.
              analysis.additional_stops.push_back("</assistant>");
              LOG_DBG(ANSI_ORANGE "[Patch: Laguna]\n" ANSI_RESET);
          }
      },
      // Bailing V3
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("Bailing V3 chat template") != std::string::npos) {
              analysis.tools.arguments.value_suffix = trim_whitespace(analysis.tools.arguments.value_suffix);
              analysis.tools.arguments.tolerate_intertag_whitespace = true;
              LOG_DBG(ANSI_ORANGE "[Patch: Bailing V3]\n" ANSI_RESET);
          }
      },

    });

// Common JSON structures
static json params_schema = {
    { "type",       "object"                                                           },
    { "properties",
     { { ARG_FIRST, { { "type", "string" }, { "description", "First argument" } } },
        { ARG_SECOND, { { "type", "string" }, { "description", "Second argument" } } } } },
    { "required",   json::array({})                                                    }
};

static json tools = json::array({
    { { "type", "function" },
     { "function",
        json{ { "name", FUN_FIRST }, { "description", "Test function foo" }, { "parameters", params_schema } } } },
    { { "type", "function" },
     { "function",
        json{ { "name", FUN_SECOND }, { "description", "Test function bar" }, { "parameters", params_schema } } } }
});

static json user_msg = json{
    { "role",    "user"  },
    { "content", USER_MSG }
};

static json build_tool_call(const std::string & name, const json & args, const std::string & id = CALL_ID_001) {
    return json{
        { "id",       id                                              },
        { "type",     "function"                                      },
        { "function", json{ { "name", name }, { "arguments", args } } }
    };
}

static json first_tool_call_zero_args         = build_tool_call(FUN_FIRST, json::object(), CALL_ID_001);
static json first_tool_call_one_arg           = build_tool_call(FUN_FIRST, {{ ARG_FIRST, "XXXX" }}, CALL_ID_001);
static json first_tool_call_one_arg_other_val = build_tool_call(FUN_FIRST, {{ ARG_FIRST, "YYYY" }}, CALL_ID_001);
static json first_tool_call_other_arg         = build_tool_call(FUN_FIRST, {{ ARG_SECOND, "YYYY" }}, CALL_ID_001);

static json first_tool_call =
    build_tool_call(FUN_FIRST, json{{ ARG_FIRST,  "XXXX" }, { ARG_SECOND, "YYYY" }}, CALL_ID_001);
static json second_tool_call =
    build_tool_call(FUN_SECOND, json{ { ARG_FIRST,  "XXXX" }, { ARG_SECOND, "YYYY" }}, CALL_ID_002);
static json first_tool_call_alt_id =
    build_tool_call(FUN_FIRST, json{{ ARG_FIRST,  "XXXX" }, { ARG_SECOND, "YYYY" }}, CALL_ID_999);

template <typename T>
static std::string mode_to_str(T mode) {
    std::ostringstream os;
    os << mode;
    return os.str();
}

void autoparser::analyze_template(const common_chat_template & tmpl) {
    jinja_caps = tmpl.original_caps();
    reasoning = analyze_reasoning(tmpl, jinja_caps.supports_tool_calls);
    content = analyze_content(tmpl, reasoning);
    tools = analyze_tools(jinja_caps.supports_tool_calls ? analyze_tools(tmpl, jinja_caps, reasoning) : analyze_tools());
    assistant_start = detect_assistant_start_marker(tmpl);
    user_start = detect_user_start_marker(tmpl);
    collect_preserved_tokens();

    for (auto & workaround : workarounds) {
        workaround(tmpl, *this);
    }

    LOG_DBG("\n--- Reasoning & Content Structure ---\n");
    LOG_DBG("user_msg_start: %s\n", user_start.c_str());
    LOG_DBG("assistant_msg_start: %s\n", assistant_start.c_str());
    LOG_DBG("reasoning_mode: %s\n", mode_to_str(reasoning.mode).c_str());
    LOG_DBG("reasoning_start: '%s'\n", reasoning.start.c_str());
    LOG_DBG("reasoning_end: '%s'\n", reasoning.end.c_str());
    LOG_DBG("content_mode: %s\n", mode_to_str(content.mode).c_str());
    LOG_DBG("content_start: '%s'\n", content.start.c_str());
    LOG_DBG("content_end: '%s'\n", content.end.c_str());

    LOG_DBG("\n--- Tool Call Structure ---\n");
    LOG_DBG("tool_mode: %s\n", mode_to_str(tools.format.mode).c_str());
    LOG_DBG("supports_tools: %s\n", jinja_caps.supports_tools ? "true" : "false");
    LOG_DBG("supports_parallel_calls: %s\n", jinja_caps.supports_parallel_tool_calls ? "true" : "false");
    LOG_DBG("tool_section_start: '%s'\n", tools.format.section_start.c_str());
    LOG_DBG("tool_section_end: '%s'\n", tools.format.section_end.c_str());
    LOG_DBG("per_call_start: '%s'\n", tools.format.per_call_start.c_str());
    LOG_DBG("per_call_end: '%s'\n", tools.format.per_call_end.c_str());
    LOG_DBG("func_name_prefix: '%s'\n", tools.function.name_prefix.c_str());
    LOG_DBG("func_name_suffix: '%s'\n", tools.function.name_suffix.c_str());
    LOG_DBG("func_args_separator: '%s'\n", tools.function.args_separator.c_str());
    LOG_DBG("func_close: '%s'\n", tools.function.close.c_str());
    LOG_DBG("call_id_prefix: '%s'\n", tools.call_id.prefix.c_str());
    LOG_DBG("call_id_suffix: '%s'\n", tools.call_id.suffix.c_str());
    LOG_DBG("call_id_pos: '%s'\n", mode_to_str(tools.call_id.pos).c_str());
    LOG_DBG("args_start: '%s'\n", tools.arguments.start.c_str());
    LOG_DBG("args_end: '%s'\n", tools.arguments.end.c_str());
    LOG_DBG("arg_name_prefix: '%s'\n", tools.arguments.name_prefix.c_str());
    LOG_DBG("arg_name_suffix: '%s'\n", tools.arguments.name_suffix.c_str());
    LOG_DBG("arg_value_prefix: '%s'\n", tools.arguments.value_prefix.c_str());
    LOG_DBG("arg_value_suffix: '%s'\n", tools.arguments.value_suffix.c_str());
    LOG_DBG("name_field: '%s'\n", tools.format.name_field.c_str());
    LOG_DBG("args_field: '%s'\n", tools.format.args_field.c_str());
    LOG_DBG("id_field: '%s'\n", tools.format.id_field.c_str());
    LOG_DBG("gen_id_field: '%s'\n", tools.format.gen_id_field.c_str());
    LOG_DBG("parameter_order: '%s'\n", std::accumulate(tools.format.parameter_order.begin(), tools.format.parameter_order.end(),
        std::string(""), [] (const std::string & a, const std::string & b) { return a.empty() ? b : a + ", " + b; }
        ).c_str());

    LOG_DBG(ANSI_PURPLE "=== Differential analysis complete ===\n" ANSI_RESET);
    analysis_complete = true;
}

void autoparser::collect_preserved_tokens() {
    auto add_token = [this](const std::string & org_token) {
        std::string token = trim_whitespace(org_token);
        if (!token.empty()) {
            // Avoid duplicates
            if (std::find(preserved_tokens.begin(), preserved_tokens.end(), token) == preserved_tokens.end()) {
                preserved_tokens.push_back(token);
            }
        }
    };

    add_token(reasoning.start);
    add_token(reasoning.end);
    add_token(content.start);
    add_token(content.end);
    add_token(tools.format.section_start);
    add_token(tools.format.section_end);
    add_token(tools.format.per_call_start);
    add_token(tools.format.per_call_end);
    add_token(tools.function.name_prefix);
    add_token(tools.function.name_suffix);
    add_token(tools.function.args_separator);
    add_token(tools.function.close);
    add_token(tools.arguments.start);
    add_token(tools.arguments.end);
    add_token(tools.arguments.name_prefix);
    add_token(tools.arguments.name_suffix);
    add_token(tools.arguments.separator);
    add_token(tools.arguments.value_prefix);
    add_token(tools.arguments.value_suffix);
    add_token(tools.call_id.prefix);
    add_token(tools.call_id.suffix);
}

std::string autoparser::detect_assistant_start_marker(const common_chat_template & tmpl) {
    json user_msg = json{
        { "role",    "user"   },
        { "content", USER_MSG }
    };

    json assistant_no_reasoning = json{
        { "role",    "assistant"   },
        { "content", ASSISTANT_MSG }
    };

    template_params params;
    params.messages              = json::array({ user_msg });
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        tmpl, params, [&](template_params & p) {
            p.messages = json::array({ user_msg, assistant_no_reasoning });
        }
    );

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed, skipping assistant start detection\n" ANSI_RESET, __func__);
        return "";
    }

    auto usermsg = comparison->diff.right;
    if (usermsg.find(ASSISTANT_MSG) == std::string::npos) {
        LOG_DBG(ANSI_ORANGE "%s: Did not find assistant message in assistant message block, skipping detection\n" ANSI_RESET, __func__);
    }

    auto ast_prefix = usermsg.substr(0, usermsg.find(ASSISTANT_MSG));
    if (!reasoning.start.empty() && ast_prefix.find(trim_whitespace(reasoning.start)) != std::string::npos) {
        ast_prefix = ast_prefix.substr(0, ast_prefix.find(trim_whitespace(reasoning.start)));
    }
    if (!reasoning.end.empty() && ast_prefix.find(trim_whitespace(reasoning.end)) != std::string::npos) {
        ast_prefix = ast_prefix.substr(0, ast_prefix.find(trim_whitespace(reasoning.end)));
    }
    return trim_whitespace(ast_prefix);
}

std::string autoparser::detect_user_start_marker(const common_chat_template & tmpl) {
    json user_msg = json{
        { "role",    "user"   },
        { "content", USER_MSG }
    };

    json assistant = json{
        { "role",    "assistant"   },
        { "content", ASSISTANT_MSG }
    };

    json user_msg_two = json{
        { "role",    "user"       },
        { "content", USER_MSG_TWO }
    };

    template_params params;
    params.messages              = json::array({});
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        tmpl, params, [&](template_params & p) {
            p.messages = json::array({ user_msg });
        }
    );

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed, unsupported empty messages? trying complex variant\n" ANSI_RESET, __func__);
        params.messages = json::array({ user_msg_two, assistant });
        comparison = compare_variants(
            tmpl, params, [&](template_params & p) {
                p.messages = json::array({ user_msg_two, assistant, user_msg });
            }
        );
        if (!comparison) {
            LOG_DBG(ANSI_ORANGE "%s: Template application failed for reserve variant, aborting\n" ANSI_RESET, __func__);
            return "";
        }
    }

    auto usermsg = comparison->diff.right;
    if (usermsg.find(USER_MSG) == std::string::npos) {
        LOG_DBG(ANSI_ORANGE "%s: Did not find user message in user message block, aborting detection\n" ANSI_RESET, __func__);
    }

    if (usermsg.find(ASSISTANT_MSG) != std::string::npos) {
        usermsg = usermsg.substr(usermsg.find(ASSISTANT_MSG) + ASSISTANT_MSG.size());
    }

    auto candidate = usermsg.substr(0, usermsg.find(USER_MSG));
    auto candidate_split = segmentize_markers(candidate);
    std::stringstream result;
    bool encountered_marker = false;
    for (const auto & mrk : candidate_split) {
        std::string lower_mrk = std::string(mrk.value);
        std::transform(lower_mrk.begin(), lower_mrk.end(), lower_mrk.begin(),
            [](unsigned char c) { return std::tolower(c); });
        // heuristic to weed out potential end markers, but only at the start
        if (mrk.type == segment_type::MARKER && !encountered_marker &&
            (lower_mrk.find("end") != std::string::npos || lower_mrk.find("close") != std::string::npos)) {
            continue;
        }
        if (mrk.type == segment_type::TEXT && !encountered_marker && trim_whitespace(mrk.value).empty()) {
            continue;
        }
        encountered_marker |= mrk.type == segment_type::MARKER;
        result << mrk.value;
    }
    return trim_whitespace(result.str());
}

analyze_reasoning::analyze_reasoning(const common_chat_template & tmpl, bool supports_tools)
    : analyze_base(tmpl) {
    LOG_DBG(ANSI_PURPLE "=== Starting differential analysis ===\n" ANSI_RESET);
    LOG_DBG(ANSI_ORANGE "Phase 1: Reasoning analysis\n" ANSI_RESET);

    compare_reasoning_presence();
    compare_thinking_enabled();
    if (supports_tools) {
        compare_reasoning_scope();
    }
}

void analyze_reasoning::compare_reasoning_presence() {
    json user_msg = json{
        { "role",    "user"  },
        { "content", USER_MSG }
    };

    json assistant_no_reasoning = json{
        { "role",    "assistant"   },
        { "content", ASSISTANT_MSG }
    };

    json assistant_with_reasoning = json{
        { "role",              "assistant"                },
        { "content",           ASSISTANT_MSG              },
        { "reasoning_content", THINKING_CONTENT           }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_no_reasoning });
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_reasoning }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed, skipping reasoning detection\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    const std::string reasoning_content = THINKING_CONTENT;

    if (!diff.right.empty() && diff.right.find(reasoning_content) != std::string::npos) {
        auto parser_delimiter = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.literal(reasoning_content) + p.space() + p.optional(p.tag("post", (p.marker() + p.space())) + p.rest());
        });
        auto parser_wrapped = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.marker() + p.space()) + p.literal(reasoning_content) + p.tag("post", (p.space() + p.marker() + p.space())) + p.rest();
        });
        // try the more aggressive parse first, if it fails, fall back to the delimiter one
        auto result = parser_wrapped.parse_anywhere_and_extract(comparison->output_B);
        if (!result.result.success()) {
            result = parser_delimiter.parse_anywhere_and_extract(comparison->output_B);
        }
        if (result.result.success()) {
            if (!result.tags["pre"].empty() && !result.tags["post"].empty()) {
                mode = reasoning_mode::TAG_BASED;
                start = result.tags["pre"];
                end   = result.tags["post"];
            } else if (!result.tags["post"].empty()) {
                mode = reasoning_mode::TAG_BASED;
                end = result.tags["post"];
            }
        }
    }
}

void analyze_reasoning::compare_thinking_enabled() {
    json user_msg = json{
        { "role",    "user"  },
        { "content", USER_MSG }
    };

    template_params params;
    params.messages              = json::array({ user_msg });
    params.add_generation_prompt = true;
    params.enable_thinking       = false;

    auto comparison = compare_variants(*tmpl, params, [&](template_params & p) { p.enable_thinking = true; });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET , __func__);
        return;
    }

    const auto & diff = comparison->diff;

    std::string left_trimmed = trim_whitespace(diff.left);
    std::string right_trimmed = trim_whitespace(diff.right);

    if (left_trimmed.empty() && !diff.right.empty()) {
        if (!right_trimmed.empty() && string_ends_with(comparison->output_B, right_trimmed)) {
            if (start.empty()) {
                start = diff.right;
                mode  = reasoning_mode::TAG_BASED;
            }
        }
    } else if (right_trimmed.empty() && !diff.left.empty()) {
        if (!left_trimmed.empty() && string_ends_with(comparison->output_A, left_trimmed)) {
            if (end.empty()) {
                auto seg = prune_whitespace_segments(segmentize_markers(comparison->output_A));
                if (seg.size() >= 2 && seg[seg.size() - 1].value == left_trimmed && seg[seg.size() - 2].type == segment_type::MARKER) {
                    start = seg[seg.size() - 2].value;
                }
                end = diff.left;
                mode = reasoning_mode::TAG_BASED;
            }
        }
    } else if (!left_trimmed.empty() && !right_trimmed.empty()) {
        // Full-output diff is noisy (e.g., SmolLM3 changes the system message when enable_thinking flips).
        // Try to find reasoning markers by tail-anchoring:
        // one output's generation prompt tail may appear in the other with extra reasoning markers appended.
        const auto & output_A = comparison->output_A;
        const auto & output_B = comparison->output_B;
        const size_t anchor_len = 64;

        for (int dir = 0; dir < 2; dir++) {
            const auto & base     = dir == 0 ? output_B : output_A;
            const auto & extended = dir == 0 ? output_A : output_B;

            size_t len = std::min(base.size(), anchor_len);
            std::string anchor = base.substr(base.size() - len);
            auto pos = extended.rfind(anchor);
            if (pos == std::string::npos || pos + len >= extended.size()) {
                continue;
            }

            std::string extra = trim_whitespace(extended.substr(pos + len));
            if (extra.empty()) {
                continue;
            }

            auto seg = prune_whitespace_segments(segmentize_markers(extra));
            if (seg.size() == 2 && seg[0].type == segment_type::MARKER && seg[1].type == segment_type::MARKER) {
                if (start.empty()) {
                    start = seg[0].value;
                }
                if (end.empty()) {
                    end   = seg[1].value;
                }
                mode = reasoning_mode::TAG_BASED;
                break;
            }
        }
    }

    if (mode == reasoning_mode::NONE && start.empty() && !end.empty()) {
        mode = reasoning_mode::TAG_BASED;
    }
}

void analyze_reasoning::compare_reasoning_scope() {
    json assistant_reasoning_content = json{
        { "role",              "assistant"      },
        { "content",           ASSISTANT_MSG    },
        { "reasoning_content", THINKING_CONTENT }
    };

    json assistant_reasoning_tools = json{
        { "role",              "assistant"                                                                  },
        { "content",           nullptr                                                                      },
        { "reasoning_content", THINKING_CONTENT                                                             },
        { "tool_calls",
         json::array({ build_tool_call(FUN_FIRST, json{ { ARG_FIRST, "VVVV" }, { ARG_SECOND, "XXXX" } }) }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_reasoning_content });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_reasoning_tools }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    std::string reasoning_content = THINKING_CONTENT;

    // Check if reasoning only appears in variant B (with tools)
    bool reasoning_in_A = comparison->output_A.find(reasoning_content) != std::string::npos;
    bool reasoning_in_B = comparison->output_B.find(reasoning_content) != std::string::npos;

    if (!reasoning_in_A && reasoning_in_B) {
        mode = reasoning_mode::TOOLS_ONLY;
        LOG_DBG(ANSI_ORANGE "%s: Detected TOOLS_ONLY reasoning mode\n" ANSI_RESET, __func__);

        auto parser_wrapped = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.marker() + p.space()) + p.literal(reasoning_content) + p.space() + p.tag("post", (p.marker() + p.space()));
        });
        auto result = parser_wrapped.parse_anywhere_and_extract(comparison->output_B);
        if (result.result.success()) {
            start = result.tags["pre"];
            end = result.tags["post"];
        } else {
            auto parser_delimiter = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                return p.literal(reasoning_content) + p.space() + p.optional(p.tag("post", (p.marker() + p.space())));
            });
            result = parser_delimiter.parse_anywhere_and_extract(comparison->output_B);
            if (result.result.success()) {
                end = result.tags["post"];
            } else {
                LOG_DBG(ANSI_ORANGE "%s: Unable to extract reasoning markers, falling back to reasoning = NONE\n" ANSI_RESET, __func__);
                mode = reasoning_mode::NONE;
            }
        }
    }
}

analyze_content::analyze_content(const common_chat_template & tmpl, const analyze_reasoning & reasoning)
    : analyze_base(tmpl) {
    LOG_DBG(ANSI_ORANGE "Phase 2: Content analysis\n" ANSI_RESET);

    json assistant_content_only = json{
        { "role",    "assistant"     },
        { "content", ASSISTANT_MSG   }
    };

    json assistant_with_tools = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ build_tool_call("test_func", json{ { "arg1", "value1" } }) }) }
    };

    json assistant_with_reasoning = json{
        { "role",              "assistant"      },
        { "content",           ""               },
        { "reasoning_content", THINKING_CONTENT }
    };

    template_params params_content_only;
    params_content_only.messages              = json::array({ user_msg, assistant_content_only });
    params_content_only.add_generation_prompt = false;
    params_content_only.enable_thinking       = true;
    params_content_only.tools                 = tools;

    auto comparison_with_tools = compare_variants(tmpl, params_content_only, [&](template_params & p) {
        p.messages = json::array({ user_msg, assistant_with_tools });
    });

    auto comparison_with_reasoning = compare_variants(tmpl, params_content_only, [&](template_params & p) {
        p.messages = json::array({ user_msg, assistant_with_reasoning });
    });

    if (!comparison_with_tools || !comparison_with_reasoning) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff_tools     = comparison_with_tools->diff;
    const auto & diff_reasoning = comparison_with_reasoning->diff;

    std::string response = ASSISTANT_MSG;

    bool found_plain_content = false;
    if (trim_whitespace(diff_tools.left) == response) {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.space() + diff_reasoning.left + p.space() + p.optional(p.marker()) + p.space() + p.end();
        });
        if (parser.parse_and_extract(diff_reasoning.left).result.success()) {
            // We only have the content text in the diff (possibly with a stray EOG marker), so no markers
            mode = content_mode::PLAIN;
            found_plain_content = true;
        } else if (reasoning.mode != reasoning_mode::NONE && !reasoning.end.empty()) {
            auto post_reasoning_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.literal(reasoning.end) + p.space() + p.literal(response);
            });
            if (post_reasoning_parser.parse_anywhere_and_extract(diff_reasoning.left).result.success()) {
                mode = content_mode::PLAIN;
                found_plain_content = true;
            }
        }
    }
    if (!found_plain_content) {
        std::string rdiff = diff_reasoning.left;
        if (!reasoning.end.empty() && rdiff.find(reasoning.end) != std::string::npos) {
            rdiff = rdiff.substr(rdiff.find(reasoning.end) + reasoning.end.length());
        }
        // Take the more promising diff
        std::string pure_content = rdiff.length() > diff_tools.left.length() ? rdiff : diff_tools.left;
        auto parser_wrapped = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.marker() + p.space()) + p.literal(response) + p.space() + p.tag("post", (p.marker() + p.space())) + p.rest();
        });
        auto result = parser_wrapped.parse_anywhere_and_extract(pure_content);
        start = result.tags["pre"];
        end = result.tags["post"];
        // TODO: WRAPPED_WITH_REASONING
    }

    // Determine content mode
    if (!start.empty() || !end.empty()) {
        mode = content_mode::ALWAYS_WRAPPED;
        // TODO: END_DELIMITED content mode - delimited at end but not at start?
    }
}

bool analyze_content::is_always_wrapped() const {
    return mode == content_mode::ALWAYS_WRAPPED && !start.empty() && !end.empty();
}

analyze_tools::analyze_tools(const common_chat_template & tmpl,
                             const jinja::caps &          caps,
                             const analyze_reasoning &    reasoning)
    : analyze_base(tmpl) {
    LOG_DBG(ANSI_ORANGE "Phase 3: Tool call analysis\n" ANSI_RESET);

    analyze_tool_calls(reasoning, caps.supports_parallel_tool_calls);

    if (format.mode != tool_format::NONE && format.mode != tool_format::JSON_NATIVE) {
        if (caps.supports_parallel_tool_calls) {
            check_per_call_markers();
        }
        LOG_DBG(ANSI_ORANGE "Phase 3a: Function call analysis\n" ANSI_RESET);
        extract_function_markers();
        LOG_DBG(ANSI_ORANGE "Phase 3b: Argument analysis\n" ANSI_RESET);
        if (format.mode == tool_format::TAG_WITH_TAGGED) {
            analyze_arguments();
        }
        extract_argument_separator();
        extract_args_markers();
        LOG_DBG(ANSI_ORANGE "Phase 3c: Call id analysis\n" ANSI_RESET);
        extract_call_id_markers();
    }
}

void analyze_tools::analyze_tool_calls(const analyze_reasoning & reasoning, bool supports_parallel_tool_calls) {
    json assistant_no_tools = json{
        { "role",    "assistant"   },
        { "content", ASSISTANT_MSG }
    };

    json assistant_with_tools = json{
        { "role",       "assistant"                      },
        { "content",    ""                               },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_no_tools });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_tools }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    std::string tool_section = diff.right;

    if (tool_section.empty()) {
        return;
    }

    analyze_tool_call_format(tool_section, FUN_FIRST, ARG_FIRST, reasoning, supports_parallel_tool_calls);
}

void analyze_tools::analyze_tool_call_format(const std::string &       haystack,
                                             const std::string &       fun_name_needle,
                                             const std::string &       arg_name_needle,
                                             const analyze_reasoning & reasoning,
                                             bool                      supports_parallel_tool_calls) {
    if (fun_name_needle.empty() || arg_name_needle.empty() || haystack.empty()) {
        return;
    }

    auto in_json_haystack = [&haystack](const std::string & needle) -> bool {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.choice({ p.literal("{"), p.literal(":") }) << p.choice({
                p.tag("dq", p.literal("\"") + p.literal(needle) + p.literal("\"")) });
        });
        auto result = parser.parse_anywhere_and_extract(haystack);
        return result.result.success();
    };

    auto fun_quote = in_json_haystack(fun_name_needle);
    auto arg_quote = in_json_haystack(arg_name_needle);

    if (fun_quote) {
        // no need to check further, we're in JSON land
        format.mode = tool_format::JSON_NATIVE;
    } else if (arg_quote) {
        format.mode = tool_format::TAG_WITH_JSON;
    } else {
        format.mode = tool_format::TAG_WITH_TAGGED;
    }

    // first, remove any reasoning markers
    std::string clean_haystack = haystack;
    if (!reasoning.start.empty()) {
        auto pos = haystack.find(reasoning.start);
        if (pos != std::string::npos) {
            clean_haystack = haystack.substr(0, pos) + haystack.substr(pos + reasoning.start.length());
        }
    }
    if (!reasoning.end.empty()) {
        auto pos = clean_haystack.find(reasoning.end);
        if (pos != std::string::npos) {
            clean_haystack = clean_haystack.substr(0, pos) + clean_haystack.substr(pos + reasoning.end.length());
        }
    }

    if (format.mode == tool_format::JSON_NATIVE) {
        analyze_tool_call_format_json_native(clean_haystack, fun_name_needle, arg_name_needle);
        if (supports_parallel_tool_calls) {
            analyze_json_native_parallel_calls();
        }
    } else {
        analyze_tool_call_format_non_json(clean_haystack, fun_name_needle);
    }
    // always relax whitespace requirements on ending markers since they don't influence content
    format.section_end  = trim_whitespace(format.section_end);
    format.per_call_end = trim_whitespace(format.per_call_end);
}

void analyze_tools::analyze_json_native_parallel_calls() {
    json assistant_one_tool = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_two_tools = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call, second_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_one_tool });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_two_tools }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    std::string & second_call = comparison->diff.right;
    if (!format.section_start.empty() && second_call.find(format.section_start) != std::string::npos) {
        format.per_call_start = format.section_start;
        format.per_call_end = format.section_end;
        format.section_start.clear();
        format.section_end.clear();
    }
}

void analyze_tools::analyze_tool_call_format_json_native(const std::string & clean_haystack,
                                                         const std::string & fun_name_needle,
                                                         const std::string & arg_name_needle) {
    // we might not have the typical OpenAI tool calling structure
    int  json_start     = clean_haystack.find_first_of('{');
    int  json_end       = clean_haystack.find_last_of('}');
    std::string cut     = clean_haystack.substr(json_start, json_end - json_start + 1);
    json call_struct    = json::parse(cut);
    auto register_field = [&](const std::string & prefix, const common_json_entry & subel) {
        if (subel.value().is_string() && std::string(subel.value()).find("call0000") != std::string::npos) {
            format.id_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.value().is_string() && std::string(subel.value()) == fun_name_needle) {
            format.name_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.value().dump().find(arg_name_needle) !=
                   std::string::npos) {  // handle both string and JSON obj variants
            format.args_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.key().find("id") != std::string::npos) {
            // heuristics for generated id field
            format.gen_id_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        }
    };
    for (const auto & el : call_struct.items()) {
        if (el.key() == fun_name_needle) {
            format.fun_name_is_key = true;
            // When function name is the key, there's no name field and args are direct
            format.name_field.clear();
            format.args_field.clear();
            // Don't register this element - the function name IS the key, not a field
        } else {
            if (el.value().is_object() &&
                el.value().dump().find(arg_name_needle) == std::string::npos) {  // not the args object
                format.function_field = el.key();
                for (const auto & subel : el.value().items()) {
                    register_field(el.key(), subel);
                }
            }
            // Register this element as a potential field
            register_field("", el);
        }
    }
    auto array_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
        return p.tag("pre", p.literal("[") + p.space()) + p.literal(cut) + p.tag("post", p.space() + p.literal("]"));
    });

    auto ar_parse_res = array_parser.parse_anywhere_and_extract(clean_haystack);
    if (ar_parse_res.result.success()) {
        format.tools_array_wrapped = true;
        json_start -= ar_parse_res.tags["pre"].length();
        json_end += ar_parse_res.tags["post"].length();
    }
    json_end++; // we want to move past the closing char for end marker extraction

    std::vector<std::pair<size_t, std::string>> located_params;
    if (!format.name_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.name_field), format.name_field });
    }
    if (!format.args_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.args_field), format.args_field });
    }
    if (!format.id_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.id_field), format.id_field });
    }
    if (!format.gen_id_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.gen_id_field), format.gen_id_field });
    }
    std::sort(located_params.begin(), located_params.end());
    for (auto & pair : located_params) {
        format.parameter_order.push_back(pair.second);
    }
    // we can immediately extract tool calling markers too
    format.section_start = trim_leading_whitespace(clean_haystack.substr(0, json_start));
    format.section_end   = trim_whitespace(clean_haystack.substr(json_end));
    // When tools_array_wrapped is true, the closing bracket is part of the array structure,
    // not a separate section end marker. Clear tool_section_end to avoid duplicate brackets.
    if (format.tools_array_wrapped && format.section_end == "]") {
        format.section_end.clear();
    }
}

void analyze_tools::analyze_tool_call_format_non_json(const std::string & clean_haystack,
                                                      const std::string & fun_name_needle) {
    // first, let's find out if the function is inside a tag or standalone
    auto fun_marker_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("fun_marker", p.choice({
            p.tag("fun_pre", p.literal("<") + p.until_one_of({ ">", fun_name_needle })) + p.literal(fun_name_needle) +
                p.tag("fun_post", p.negate(p.space() + p.literal("<")) + p.until(">") + p.literal(">")) + p.space(),
            p.tag("fun_pre", p.literal("[") + p.until_one_of({ "]", fun_name_needle })) + p.literal(fun_name_needle) +
                p.tag("fun_post", p.negate(p.space() + p.literal("[") + p.until("]") + p.literal("]")) + p.space()) }));
    });
    auto fun_res = fun_marker_parser.parse_anywhere_and_extract(clean_haystack);
    std::string fun_marker = fun_name_needle;
    if (fun_res.result.success()) {
        fun_marker = fun_res.tags["fun_marker"];
    }
    // now, consume up to two markers, then treat everything up to the function marker as function name prefix
    auto per_tool_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
        return p.tag("sec_start", p.marker() + p.space()) + p.tag("call_start", p.marker() + p.space()) +
            p.tag("fun_pre", p.until(fun_marker)) + fun_marker + p.tag("rest", p.rest());
    });
    auto section_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
        return p.tag("sec_start", p.marker() + p.space()) + fun_marker + p.tag("rest", p.rest());
    });
    auto result = per_tool_parser.parse_anywhere_and_extract(clean_haystack);
    tagged_parse_result result_end;
    if (result.result.success()) {
        auto double_closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("call_end", p.marker() + p.space()) + p.tag("sec_end", p.marker() + p.space()) + p.end();
        });
        result_end = double_closer_parser.parse_anywhere_and_extract(result.tags["rest"]);
        function.name_prefix = fun_res.tags["fun_pre"] + function.name_prefix;
    } else {
        result = section_parser.parse_anywhere_and_extract(clean_haystack);
        auto single_closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("sec_end", p.marker() + p.space()) + p.end();
        });
        result_end = single_closer_parser.parse_anywhere_and_extract(result.tags["rest"]);
    }
    format.per_call_start = result.tags["call_start"];
    format.per_call_end = result_end.tags["call_end"];
    format.section_start = result.tags["sec_start"];
    format.section_end = result_end.tags["sec_end"];
}

void analyze_tools::check_per_call_markers() {
    json assistant_one_tool = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_two_tools = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call, second_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_one_tool });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto one_vs_two = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_two_tools }); });

    if (!one_vs_two) {
        LOG_DBG(ANSI_ORANGE "%s: Generating double tool call comparison failed\n" ANSI_RESET, __func__);
        return;
    }

    diff_split filter_common_call_part = calculate_diff_split(one_vs_two->diff.suffix, one_vs_two->diff.right);

    std::string second_tool_content = trim_leading_whitespace(filter_common_call_part.right);
    if (!format.section_start.empty() &&
        second_tool_content.find(format.section_start) == 0) {
        format.per_call_start = format.section_start;
        format.per_call_end   = format.section_end;
        format.section_start.clear();
        format.section_end.clear();
    }

    if (!format.per_call_end.empty()) {
        auto count_occurrences = [](const std::string & haystack, const std::string & needle) {
            size_t count = 0;
            for (size_t pos = haystack.find(needle); pos != std::string::npos;
                 pos = haystack.find(needle, pos + needle.size())) {
                count++;
            }
            return count;
        };
        size_t calls_one = count_occurrences(one_vs_two->output_A, format.per_call_end);
        size_t calls_two = count_occurrences(one_vs_two->output_B, format.per_call_end);
        if (calls_one > 0 && calls_one == calls_two) {
            format.section_end = format.per_call_end;
            format.per_call_end.clear();
        }
    }
}

void analyze_tools::extract_function_markers() {
    json assistant_nocall = json{
        { "role",    "assistant"   },
        { "content", ASSISTANT_MSG },
    };

    json assistant_foofoo = json{
        { "role",       "assistant"                      },
        { "content",    ""                               },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_barbar = json{
        { "role",       "assistant"                       },
        { "content",    ""                                },
        { "tool_calls", json::array({ second_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_foofoo });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_barbar }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (diff.left.find(FUN_FIRST) != std::string::npos && diff.right.find(FUN_SECOND) != std::string::npos) {
        std::string prefix_marker;
        if (!format.per_call_start.empty()) {
            prefix_marker = format.per_call_start;
        } else {
            prefix_marker = format.section_start;
        }
        if (!prefix_marker.empty() && diff.prefix.rfind(prefix_marker) != std::string::npos) {
            function.name_prefix =
                diff.prefix.substr(diff.prefix.rfind(prefix_marker) + prefix_marker.size());
        }

        // Extract name prefix/suffix from diff.left (stop at the next marker boundary)
        auto name_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.until(FUN_FIRST)) + p.literal(FUN_FIRST) +
                   p.tag("post", p.zero_or_more(p.negate(p.marker()) + p.any()));
        });
        auto name_result = name_parser.parse_and_extract(diff.left);
        if (name_result.result.success()) {
            function.name_prefix += name_result.tags["pre"];
            function.name_suffix = name_result.tags["post"];
        }

        // Extend name_suffix with content from diff.suffix before args begin
        if (format.mode == tool_format::TAG_WITH_JSON) {
            // For JSON: name_suffix extends to the first non-marker { or [, including any
            // markers along the way. Only applies if there's at least one marker after
            // the JSON content (matching the original "stop < seg_suf.size() - 1" guard).
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                auto non_json = p.marker() | (p.negate(p.literal("{")) + p.negate(p.literal("[")) + p.any());
                auto after_json = p.zero_or_more(p.negate(p.marker()) + p.any()) + p.marker();
                return p.tag("ext", p.zero_or_more(non_json)) + after_json;
            });
            auto suf_result = suffix_parser.parse_and_extract(diff.suffix);
            if (suf_result.result.success()) {
                function.name_suffix += suf_result.tags["ext"];
            }
        } else {
            // For tagged: name_suffix extends to the first marker (arg marker)
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                return p.tag("ext", p.zero_or_more(p.negate(p.marker()) + p.any()));
            });
            auto suf_result = suffix_parser.parse_and_extract(diff.suffix);
            if (suf_result.result.success()) {
                function.name_suffix += suf_result.tags["ext"];

                auto arg_start = [&](common_peg_parser_builder &p) {
                    return p.marker() + p.space() + p.choice({ p.literal(ARG_FIRST), p.literal(ARG_SECOND) });
                };
                auto sep_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                    return p.tag("sep", p.zero_or_more(p.negate(arg_start(p)) + p.any())) + arg_start(p);
                });
                auto sep_result = sep_parser.parse_and_extract(diff.suffix.substr(suf_result.tags["ext"].size()));
                if (sep_result.result.success()) {
                    function.args_separator = trim_whitespace(sep_result.tags["sep"]);
                }
            }
        }

        // Extract the closer (between last arg and call/section end marker)
        std::string suffix_marker;
        if (!format.per_call_end.empty()) {
            suffix_marker = format.per_call_end;
        } else {
            suffix_marker = format.section_end;
        }
        std::string closer_suffix;
        if (suffix_marker.empty()) {
            // we'll have to rely on an extra diff with no-calls version
            auto notool_comp = compare_variants(
                *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_nocall }); });
            if (notool_comp) {
                auto nt_diff  = notool_comp->diff;
                closer_suffix = nt_diff.left.substr(nt_diff.left.find("YYYY") + 4);
            }
        } else {
            closer_suffix = diff.suffix.substr(0, diff.suffix.find(suffix_marker));
        }
        if (!closer_suffix.empty()) {
            if (format.mode == tool_format::TAG_WITH_TAGGED) {
                // After last arg value, skip the closing arg marker, rest is closer
                auto closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                    return p.until("YYYY") + p.literal("YYYY") + p.space() +
                           p.marker() + p.space() +
                           p.tag("close", p.rest());
                });
                auto close_result = closer_parser.parse_and_extract(closer_suffix);
                if (close_result.result.success()) {
                    function.close = close_result.tags["close"];
                }
            } else if (format.mode == tool_format::TAG_WITH_JSON) {
                // After last arg value, find end of JSON args, rest is closer
                auto closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                    return p.until("YYYY") + p.literal("YYYY") + p.tag("post_val", p.rest());
                });
                auto close_result = closer_parser.parse_and_extract(closer_suffix);
                if (close_result.result.success()) {
                    const auto & post = close_result.tags["post_val"];
                    size_t pos = post.find_last_of("}]");
                    if (pos != std::string::npos && pos < post.size() - 1) {
                        function.close = trim_leading_whitespace(post.substr(pos + 1));
                    }
                }
            }
        }
        function.close = trim_leading_whitespace(function.close);
    }
}

void analyze_tools::analyze_arguments() {
    extract_argument_name_markers();
    extract_argument_value_markers();
}

void analyze_tools::extract_argument_name_markers() {
    json assistant_first_arg = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    json assistant_second_arg = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_other_arg }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_first_arg });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_second_arg }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (!diff.left.empty() && !diff.right.empty()) {
        // Parse both sides to find ARG_FIRST/ARG_SECOND and extract the surrounding structure
        auto left_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.tag("pre", p.until(ARG_FIRST)) + p.literal(ARG_FIRST) +
                   p.tag("suffix", p.until_one_of({"\"", "X"}));
        });
        auto right_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.tag("pre", p.until(ARG_SECOND)) + p.literal(ARG_SECOND) +
                   p.tag("suffix", p.until_one_of({"\"", "Y"}));
        });
        auto left_result  = left_parser.parse_anywhere_and_extract(diff.left);
        auto right_result = right_parser.parse_anywhere_and_extract(diff.right);

        if (left_result.result.success() && right_result.result.success() &&
            !left_result.tags["pre"].empty() &&
            left_result.tags["pre"] == right_result.tags["pre"] &&
            left_result.tags["suffix"] == right_result.tags["suffix"]) {
            // Name is inside a structure (e.g., JSON key): prefix is the shared wrapper
            arguments.name_prefix = left_result.tags["pre"];
            arguments.name_suffix = left_result.tags["suffix"];
        } else if (diff.left.substr(0, ARG_FIRST.length()) == ARG_FIRST && diff.right.substr(0, ARG_SECOND.length()) == ARG_SECOND) {
            // Name is directly in the diff: prefix comes from last marker in diff.prefix
            auto pre_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                auto last_marker = p.marker() + p.zero_or_more(p.negate(p.marker()) + p.any()) + p.end();
                return p.zero_or_more(p.negate(last_marker) + p.any()) + p.tag("name_prefix", last_marker);
            });
            auto pre_result = pre_parser.parse_and_extract(diff.prefix);
            arguments.name_prefix = pre_result.result.success()
                ? pre_result.tags["name_prefix"] : diff.prefix;

            // Suffix extends from after ARG_FIRST to the first marker (+ optional whitespace).
            // The marker could be in diff.left itself or in diff.suffix, so we concatenate.
            std::string after_first = diff.left.substr(ARG_FIRST.length()) + diff.suffix;
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.tag("suffix", p.zero_or_more(p.negate(p.marker()) + p.any()) +
                                       p.marker() + p.space());
            });
            auto suf_result = suffix_parser.parse_anywhere_and_extract(after_first);
            if (suf_result.result.success()) {
                arguments.name_suffix = suf_result.tags["suffix"];
            }
        }
    }
}

void analyze_tools::extract_argument_value_markers() {
    json assistant_val_X = json{
        { "role",       "assistant"                              },
        { "content",    ""                                       },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    json assistant_val_Y = json{
        { "role",       "assistant"                                        },
        { "content",    ""                                                 },
        { "tool_calls", json::array({ first_tool_call_one_arg_other_val }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_val_X });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_val_Y }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (diff.left == "XXXX" && diff.right == "YYYY") {
        std::string arg_name_ending = ARG_FIRST + arguments.name_suffix;
        std::string prefix          = diff.prefix;
        if (prefix.rfind(arg_name_ending) != std::string::npos) {
            prefix = prefix.substr(prefix.rfind(arg_name_ending) + arg_name_ending.size());
        }
        if (!prefix.empty()) {
            // Find the last marker + any trailing non-marker text to end
            auto prefix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                auto last_marker = p.marker() + p.zero_or_more(p.negate(p.marker()) + p.any()) + p.end();
                return p.zero_or_more(p.negate(last_marker) + p.any()) + p.tag("val_prefix", last_marker);
            });
            auto pre_result = prefix_parser.parse_and_extract(prefix);
            arguments.value_prefix = pre_result.result.success() ? pre_result.tags["val_prefix"] : prefix;
        }

        std::string value_suffix = diff.suffix;
        if (!function.close.empty()) {
            size_t func_close_pos = value_suffix.find(function.close);
            if (func_close_pos != std::string::npos) {
                value_suffix = value_suffix.substr(0, func_close_pos);
            }
        } else if (!format.per_call_end.empty() || !format.section_end.empty()) {
            std::string end_marker =
                !format.per_call_end.empty() ? format.per_call_end : format.section_end;
            size_t end_marker_pos = value_suffix.find(end_marker);
            if (end_marker_pos != std::string::npos) {
                value_suffix = value_suffix.substr(0, end_marker_pos);
            }
        }
        if (!trim_whitespace(value_suffix).empty()) {
            arguments.value_suffix = value_suffix;
        }
    }
}

void analyze_tools::extract_argument_separator() {
    json assistant_one_arg = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    json assistant_two_args = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_one_arg });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_two_args }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (!diff.right.empty()) {
        std::string separator        = until_common_prefix(diff.right, ARG_FIRST, ARG_SECOND);
        arguments.separator = separator;
    }
}

void analyze_tools::extract_args_markers() {
    json assistant_no_args = json{
        { "role",       "assistant"},
        { "content",    ""         },
        { "tool_calls", json::array({ first_tool_call_zero_args }) }
    };

    json assistant_with_args = json{
        { "role",       "assistant"},
        { "content",    ""         },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_no_args });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_args }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (format.mode == tool_format::JSON_NATIVE) {
        std::string prefix_marker = !format.section_start.empty() ? format.section_start : format.per_call_start;
        std::string suffix_marker = !format.section_end.empty() ? format.section_end : format.per_call_end;
        // these might happen earlier in the tools section as an example or somewhere else, so we need to find the closest ones
        size_t prefix_pos = prefix_marker.empty() ? 0 : diff.prefix.rfind(prefix_marker);
        size_t suffix_pos = suffix_marker.empty() ? diff.suffix.size() : diff.suffix.find(suffix_marker);
        if (prefix_pos == std::string::npos) {
            prefix_pos = 0;
        }
        if (suffix_pos == std::string::npos) {
            suffix_pos = diff.suffix.size();
        }
        std::string prefix_cut = diff.prefix.substr(prefix_pos + prefix_marker.size());
        std::string suffix_cut = diff.suffix.substr(0, suffix_pos);
        std::string args_start = until_common_prefix(prefix_cut, "{}", "{\"first\":");
        std::string args_end   = after_common_suffix(suffix_cut, "{}", "\"XXXX\"}");

        if (!args_start.empty() || !args_end.empty()) {
            size_t find_fun = args_start.find(FUN_FIRST);
            if (find_fun != std::string::npos) {
                args_start = args_start.substr(find_fun + FUN_FIRST.size(), args_start.size() - find_fun - FUN_FIRST.size());
            }
            size_t find_call_id = args_start.find(CALL_ID_001);
            if (find_call_id != std::string::npos) {
                args_start = args_start.substr(find_call_id + CALL_ID_001.size(), args_start.size() - find_call_id - CALL_ID_001.size());
            }
            arguments.start = args_start;
            arguments.end   = args_end;
        }
    }
}

void analyze_tools::extract_call_id_markers() {
    json assistant_id1 = json{
        { "role",       "assistant" },
        { "content",    ""                               },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_id2 = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_alt_id }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_id1 });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_id2 }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed for call_id detection\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (diff.left.empty() && diff.right.empty()) {
        return;
    }

    std::string id_value_1 = CALL_ID_001;
    std::string id_value_2 = CALL_ID_999;

    size_t common_id_prefix_len = 0;
    for (size_t i = 0; i < std::min(id_value_1.length(), id_value_2.length()); i++) {
        if (id_value_1[i] == id_value_2[i]) {
            common_id_prefix_len++;
        } else {
            break;
        }
    }
    std::string common_id_part = id_value_1.substr(0, common_id_prefix_len);

    // Check if the function name is in the prefix (normal case: BETWEEN_FUNC_AND_ARGS or POST_ARGS)
    // or in the suffix (call_id is PRE_FUNC_NAME)
    std::string func_name           = FUN_FIRST;
    size_t      func_name_in_prefix = diff.prefix.rfind(func_name);
    size_t      func_name_in_suffix = diff.suffix.find(func_name);

    // Helper: find the last marker in a string (returns just the marker, not trailing text)
    auto find_last_marker = [](const std::string & str) -> std::string {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            auto last = p.marker() + p.zero_or_more(p.negate(p.marker()) + p.any()) + p.end();
            return p.zero_or_more(p.negate(last) + p.any()) + p.tag("m", p.marker());
        });
        auto res = parser.parse_anywhere_and_extract(str);
        return res.result.success() ? res.tags["m"] : "";
    };

    // Helper: find the first marker in a string
    auto find_first_marker = [](const std::string & str) -> std::string {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.tag("m", p.marker());
        });
        auto res = parser.parse_anywhere_and_extract(str);
        return res.result.success() ? res.tags["m"] : "";
    };

    if (func_name_in_prefix != std::string::npos && func_name_in_suffix == std::string::npos) {
        // Function name is only in prefix - call_id is BETWEEN_FUNC_AND_ARGS or POST_ARGS
        // Check if args indicator "{" is in prefix or suffix
        size_t args_in_prefix = diff.prefix.find('{', func_name_in_prefix);
        size_t args_in_suffix = diff.suffix.find('{');

        if (args_in_suffix != std::string::npos &&
            (args_in_prefix == std::string::npos || args_in_prefix > diff.prefix.length())) {
            // Args are in suffix, so call_id is BETWEEN_FUNC_AND_ARGS
            call_id.pos = call_id_position::BETWEEN_FUNC_AND_ARGS;

            // Find call_id_prefix: marker immediately preceding common_id_part (no intervening markers)
            std::string after_func = diff.prefix.substr(func_name_in_prefix + func_name.length());
            auto id_prefix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.tag("prefix", p.marker()) +
                       p.zero_or_more(p.negate(p.marker()) + p.negate(p.literal(common_id_part)) + p.any()) +
                       p.literal(common_id_part);
            });
            auto id_res = id_prefix_parser.parse_anywhere_and_extract(after_func);
            if (id_res.result.success()) {
                call_id.prefix = id_res.tags["prefix"];
            } else {
                // Fallback: use the last marker in after_func
                call_id.prefix = find_last_marker(after_func);
            }

            // Extract call_id_suffix: the first marker in the suffix before args "{"
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.zero_or_more(p.negate(p.marker()) + p.negate(p.literal("{")) + p.any()) +
                       p.tag("suffix", p.marker());
            });
            auto suf_res = suffix_parser.parse_anywhere_and_extract(diff.suffix);
            if (suf_res.result.success()) {
                call_id.suffix = suf_res.tags["suffix"];
            }
        } else if (args_in_prefix != std::string::npos) {
            // Args are in prefix, so call_id is POST_ARGS
            call_id.pos = call_id_position::POST_ARGS;

            // Extract last marker between args closing brace and the ID
            std::string after_args    = diff.prefix.substr(args_in_prefix);
            size_t      closing_brace = after_args.rfind('}');
            if (closing_brace != std::string::npos) {
                std::string between_args_and_id = after_args.substr(closing_brace + 1);
                call_id.prefix = find_last_marker(between_args_and_id);
            }

            // call_id_suffix: first marker in diff.suffix
            call_id.suffix = find_first_marker(diff.suffix);
        }
    } else if (func_name_in_suffix != std::string::npos && func_name_in_prefix == std::string::npos) {
        // Function name is only in suffix - call_id is PRE_FUNC_NAME
        call_id.pos = call_id_position::PRE_FUNC_NAME;

        // call_id_prefix: last marker in diff.prefix
        call_id.prefix = find_last_marker(diff.prefix);

        // call_id_suffix: first marker in the portion of diff.suffix before func_name
        std::string before_func = diff.suffix.substr(0, func_name_in_suffix);
        call_id.suffix = find_first_marker(before_func);
    }

    if (call_id.prefix == arguments.end) {
        call_id.prefix = "";
    }

    if (call_id.suffix == arguments.start) {
        call_id.suffix = "";
    }

    // When call_id is detected, per_call_end may have been incorrectly set to include
    // the call_id_suffix and sample args. Clear it if it starts with call_id_suffix.
    if (call_id.pos != call_id_position::NONE && !call_id.suffix.empty() &&
        format.per_call_end.find(call_id.suffix) == 0) {
        format.per_call_end.clear();
    }
}

}  // namespace autoparser

// file: common/chat-peg-parser.cpp
#include "chat-peg-parser.h"

#include "chat-auto-parser.h"
#include "ggml.h"
#include "peg-parser.h"

#include <cstdint>
#include <functional>

using ordered_json = common_json;

static std::string_view trim_trailing_space(std::string_view sv, int max = -1) {
    int count = 0;
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        if (max != -1 && count >= max) {
            break;
        }
        sv.remove_suffix(1);
        count++;
    }
    return sv;
}

static std::string_view trim_leading_space(std::string_view sv, int max = -1) {
    int count = 0;
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
        if (max != -1 && count >= max) {
            break;
        }
        sv.remove_prefix(1);
        count++;
    }
    return sv;
}

static std::string_view trim(std::string_view sv) {
    return trim_trailing_space(trim_leading_space(sv, 1));
}

// Count the number of unclosed '{' braces in a JSON-like string,
// properly skipping braces inside quoted strings.
static int json_brace_depth(const std::string & s) {
    int  depth     = 0;
    bool in_string = false;
    bool escaped   = false;
    for (char c : s) {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\' && in_string) {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (!in_string) {
            if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
            }
        }
    }
    return depth;
}

// JSON-escape a string and return the inner content (without surrounding quotes).
static std::string escape_json_string_inner(const std::string & s) {
    std::string escaped = ordered_json(s).dump();
    if (escaped.size() >= 2 && escaped.front() == '"' && escaped.back() == '"') {
        return escaped.substr(1, escaped.size() - 2);
    }
    return escaped;
}

// Convert Python-style single-quoted strings to JSON double-quoted strings
// Only converts outer string delimiters, properly handling escape sequences:
// - {'key': 'value'} -> {"key": "value"}
// - {'code': 'print(\'hello\')'} -> {"code": "print('hello')"}
// - {'msg': 'He said "hi"'} -> {"msg": "He said \"hi\""}
static std::string normalize_quotes_to_json(const std::string & input) {
    std::string result;
    result.reserve(input.size() + 16);  // May need extra space for escaping

    bool in_single_quoted = false;
    bool in_double_quoted = false;

    auto is_word_char = [](char ch) { return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_'; };

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        // Handle escape sequences
        if (c == '\\' && i + 1 < input.size()) {
            char next = input[i + 1];

            if (in_single_quoted) {
                // Inside a single-quoted string being converted to double quotes
                if (next == '\'') {
                    // \' -> ' (escaped single quote becomes unescaped in double-quoted string)
                    result += '\'';
                    ++i;
                    continue;
                }
                if (next == '"') {
                    // \" stays as \" (already escaped, works in double-quoted string)
                    result += "\\\"";
                    ++i;
                    continue;
                }
                // Other escapes (\n, \\, etc.): pass through both characters
                result += c;
                result += next;
                ++i;
                continue;
            }

            if (in_double_quoted) {
                // Inside a double-quoted string - pass through escape sequences as-is
                result += c;
                result += next;
                ++i;
                continue;
            }

            // Outside any string - just pass through the backslash
            result += c;
            continue;
        }

        // Handle quote characters
        if (c == '"') {
            if (in_single_quoted) {
                // Unescaped double quote inside single-quoted string -> must escape for JSON
                result += "\\\"";
            } else {
                // Double quote as string delimiter or outside strings
                in_double_quoted = !in_double_quoted;
                result += c;
            }
        } else if (c == '\'') {
            if (in_double_quoted) {
                // Single quote inside double-quoted string -> pass through
                result += c;
            } else if (in_single_quoted) {
                // Closing single quote -> convert to double quote
                in_single_quoted = false;
                result += '"';
            } else {
                // Opening single quote -> convert to double quote
                in_single_quoted = true;
                result += '"';
            }
        } else if (!in_single_quoted && !in_double_quoted && (c == 'T' || c == 'F' || c == 'N') &&
                   (i == 0 || !is_word_char(input[i - 1]))) {
            // Python literals -> JSON; prefix match keeps streamed partials monotonic.
            static constexpr std::pair<std::string_view, std::string_view> literals[] = {
                { "True", "true" }, { "False", "false" }, { "None", "null" },
            };
            size_t n = 0;
            while (i + n < input.size() && is_word_char(input[i + n])) {
                ++n;
            }
            std::string_view token(input.data() + i, n);
            bool matched = false;
            for (const auto & [py, js] : literals) {
                if (py.substr(0, n) == token) {
                    result += js.substr(0, n);
                    i += n - 1;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                result += c;
            }
        } else {
            result += c;
        }
    }

    return result;
}

void tag_based_peg_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    arena.visit(result, [this](const common_peg_ast_node & node) {
        if (!node.tag.empty()) {
            tags[node.tag] = std::string(node.text);
        }
    });
}

tagged_parse_result tagged_peg_parser::parse_and_extract(const std::string & input, common_peg_parse_flags extra_flags) const {
    common_peg_parse_context ctx(input, flags | extra_flags);
    auto parse_result = arena.parse(ctx);

    tag_based_peg_mapper mapper;
    mapper.from_ast(ctx.ast, parse_result);

    return { std::move(parse_result), std::move(mapper.tags) };
}

tagged_parse_result tagged_peg_parser::parse_anywhere_and_extract(const std::string & input) const {
    if (input.empty()) {
        return parse_and_extract(input);
    }
    for (size_t i = 0; i < input.size(); i++) {
        common_peg_parse_context ctx(input, flags);
        auto parse_result = arena.parse(ctx, i);
        if (parse_result.success() || i == input.size() - 1) {
            tag_based_peg_mapper mapper;
            mapper.from_ast(ctx.ast, parse_result);
            return { std::move(parse_result), std::move(mapper.tags) };
        }
    }
    GGML_ABORT("Should not happen");
}

tagged_peg_parser build_tagged_peg_parser(
    const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn) {
    common_peg_parser_builder builder;
    builder.set_root(fn(builder));
    return { builder.build() };
}

common_peg_parser common_chat_peg_builder::tag_with_safe_content(const std::string &       tag_name,
                                                                 const std::string &       marker,
                                                                 const common_peg_parser & p) {
    if (marker.empty()) {
        return zero_or_more(choice({ p, rule(tag_name, content(any())) }));
    }
    auto content_chunk = rule(tag_name, content(negate(literal(marker)) + any() + until(marker)));
    return zero_or_more(choice({ p, content_chunk }));
}

common_peg_parser common_chat_peg_builder::permute(const std::string &                    rule_prefix,
                                                   const std::vector<common_peg_parser> & parsers) {
    if (parsers.empty()) {
        return eps();
    }

    if (parsers.size() == 1 || parsers.size() > COMMON_CHAT_MAX_PERMUTE) {
        return sequence(parsers);
    }

    std::map<uint32_t, common_peg_parser>      rules;
    std::function<common_peg_parser(uint32_t)> remaining_of;

    remaining_of = [&](uint32_t remaining) -> common_peg_parser {
        if (remaining == 0) {
            return eps();
        }

        auto cached = rules.find(remaining);
        if (cached != rules.end()) {
            return cached->second;
        }

        auto alternatives = choice();
        for (size_t i = 0; i < parsers.size(); i++) {
            const uint32_t bit = 1u << i;
            if (remaining & bit) {
                alternatives |= parsers[i] + remaining_of(remaining & ~bit);
            }
        }

        return rules.emplace(remaining, rule(rule_prefix + "-" + std::to_string(remaining), alternatives)).first->second;
    };

    return remaining_of((1u << parsers.size()) - 1);
}

std::string & common_chat_peg_mapper::args_target() {
    return (current_tool && !current_tool->name.empty()) ? current_tool->arguments : args_buffer;
}

std::string common_chat_peg_mapper::normalize_container_value(const std::string & input) {
    return normalize_quotes_to_json(input);
}

void common_chat_peg_mapper::from_ast(const common_peg_ast_arena &    arena,
                                      const common_peg_parse_result & parse_result_arg) {
    arena.visit(parse_result_arg, [this](const common_peg_ast_node & node) { map(node); });
    // Flush any pending tool call that was started but never got a name
    // This happens during partial parsing when the tool call is incomplete
    if (pending_tool_call.has_value() && !pending_tool_call->name.empty()) {
        if (!args_buffer.empty()) {
            pending_tool_call->arguments = args_buffer;
        }
        if (closing_quote_pending && !pending_tool_call->arguments.empty()) {
            pending_tool_call->arguments += "\"";
        }
        result.tool_calls.push_back(pending_tool_call.value());
        pending_tool_call.reset();
    }

    // Discard whitespace-only reasoning content (e.g. from <think></think> prefill)
    if (!result.reasoning_content.empty()) {
        bool all_whitespace = true;
        for (char c : result.reasoning_content) {
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                all_whitespace = false;
                break;
            }
        }
        if (all_whitespace) {
            result.reasoning_content.clear();
        }
    }
}

void common_chat_peg_mapper::map(const common_peg_ast_node & node) {
    // Handle reasoning/content tags
    bool is_reasoning = node.tag == common_chat_peg_builder::REASONING;
    bool is_content   = node.tag == common_chat_peg_builder::CONTENT;

    if (is_reasoning) { // GPT OSS can have more than 1 reasoning block, so concatenate here
        result.reasoning_content += node.sanitized_text();
    }

    if (is_content) {
        // Concatenate content from multiple content nodes (e.g., when reasoning markers
        // are preserved before content markers in reasoning_format=NONE mode)
        result.content += node.sanitized_text();
    }

    // Handle tool-related tags (supporting both JSON and tagged formats)
    bool is_tool_open  = node.tag == common_chat_peg_builder::TOOL_OPEN;
    bool is_tool_close = node.tag == common_chat_peg_builder::TOOL_CLOSE;
    bool is_tool_name  = node.tag == common_chat_peg_builder::TOOL_NAME;
    bool is_tool_id    = node.tag == common_chat_peg_builder::TOOL_ID;
    bool is_tool_args  = node.tag == common_chat_peg_builder::TOOL_ARGS;
    bool is_arg_open   = node.tag == common_chat_peg_builder::TOOL_ARG_OPEN;
    bool is_arg_close  = node.tag == common_chat_peg_builder::TOOL_ARG_CLOSE;
    bool is_arg_name         = node.tag == common_chat_peg_builder::TOOL_ARG_NAME;
    bool is_arg_value        = node.tag == common_chat_peg_builder::TOOL_ARG_VALUE;
    bool is_arg_string_value = node.tag == common_chat_peg_builder::TOOL_ARG_STRING_VALUE;

    if (is_tool_open) {
        pending_tool_call     = common_chat_tool_call();
        current_tool          = &pending_tool_call.value();
        arg_count             = 0;
        args_buffer.clear();
        closing_quote_pending = false;
    }

    if (is_tool_id && current_tool) {
        auto text = trim_trailing_space(node.text);
        if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
            text = text.substr(1, text.size() - 2);
        }
        current_tool->id = std::string(text);
    }

    if (is_tool_name && current_tool) {
        current_tool->name = std::string(trim_trailing_space(node.text));
        // Now that we have the name, populate the arguments from the buffer
        if (!args_buffer.empty()) {
            current_tool->arguments = args_buffer;
            args_buffer.clear();
        } else if (current_tool->arguments.empty()) {
            current_tool->arguments = "{";
        }
        // Add the tool call to results so streaming can see it
        if (pending_tool_call.has_value()) {
            result.tool_calls.push_back(pending_tool_call.value());
            pending_tool_call.reset();
            current_tool = &result.tool_calls.back();
        }
    }

    if (is_tool_args && current_tool) {
        // For JSON format: arguments come as a complete JSON object
        // For tagged format: built up from individual arg_name/arg_value nodes
        auto text = trim_trailing_space(node.text);
        if (!text.empty() && text.front() == '{') {
            args_target() = std::string(text);
        }
    }

    if (is_arg_open) {
        closing_quote_pending = false;
    }

    if (is_arg_name && current_tool) {
        std::string arg_entry;
        if (arg_count > 0) {
            arg_entry = ",";
        }
        arg_entry += ordered_json(trim(node.text)).dump() + ":";
        ++arg_count;

        auto & target = args_target();
        if (target.empty()) {
            target = "{";
        }
        target += arg_entry;
    }

    if ((is_arg_value || is_arg_string_value) && current_tool) {
        std::string value_content = std::string(node.text);

        std::string value_to_add;
        if (value_content.empty() && is_arg_string_value) {
            // Empty string value - arg_close will add the closing quote
            value_to_add          = "\"";
            closing_quote_pending = true;
        } else if (!value_content.empty() && is_arg_string_value) {
            // Schema declares this as string type - always treat as literal string value
            if (!closing_quote_pending) {
                value_to_add          = "\"";
                closing_quote_pending = true;
            }
            value_to_add += escape_json_string_inner(value_content);
        } else if (!value_content.empty()) {
            // Pythonic scalars/containers -> JSON.
            value_to_add += normalize_container_value(value_content);
        }

        args_target() += value_to_add;
    }

    if (is_arg_close && current_tool) {
        if (closing_quote_pending) {
            args_target() += "\"";
            closing_quote_pending = false;
        }
    }

    if (is_tool_close && current_tool) {
        // Flush buffer to arguments if tool name was never seen
        if (current_tool->name.empty() && !args_buffer.empty()) {
            current_tool->arguments = args_buffer;
            args_buffer.clear();
        }
        // Close any pending string quote
        if (closing_quote_pending) {
            current_tool->arguments += "\"";
            closing_quote_pending = false;
        }
        // Close any unclosed braces (accounts for nested objects)
        for (int d = json_brace_depth(current_tool->arguments); d > 0; d--) {
            current_tool->arguments += "}";
        }
        // Add tool call to results if named; otherwise discard
        if (pending_tool_call.has_value()) {
            if (!current_tool->name.empty()) {
                result.tool_calls.push_back(pending_tool_call.value());
            }
            pending_tool_call.reset();
        }
    }
}

common_peg_parser common_chat_peg_builder::standard_constructed_tools(
    const std::map<std::string, std::string> & markers,
    const ordered_json &                       tools,
    bool                                       parallel_tool_calls,
    bool                                       force_tool_calls) {
    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    // Extract markers with defaults
    auto get_marker = [&markers](const std::string & key, const std::string & default_val = "") -> std::string {
        auto it = markers.find(key);
        return it != markers.end() ? it->second : default_val;
    };

    std::string section_start    = get_marker("tool_call_start_marker", "<tool_call>");
    std::string section_end      = get_marker("tool_call_end_marker", "</tool_call>");
    std::string func_opener      = get_marker("function_opener", "<function=");
    std::string func_name_suffix = get_marker("function_name_suffix", ">");
    std::string func_closer      = get_marker("function_closer", "</function>");
    std::string param_key_prefix = get_marker("parameter_key_prefix", "<param=");
    std::string param_key_suffix = get_marker("parameter_key_suffix", ">");
    std::string param_closer     = get_marker("parameter_closer", "</param>");

    // Build tool choices for tagged format
    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = common_chat_tool_parameters(function);

        // Build argument parsers
        auto args = eps();
        if (params.contains("properties") && !params["properties"].empty()) {
            auto arg_choice = choice();
            for (const auto & el : params["properties"].items()) {
                const std::string & prop_name = el.key();

                auto arg_name_parser =
                    choice({ literal(prop_name), literal("\"" + prop_name + "\""), literal("'" + prop_name + "'") });

                auto arg_rule = tool_arg(tool_arg_open(literal(param_key_prefix)) + tool_arg_name(arg_name_parser) +
                                         literal(param_key_suffix) + tool_arg_value(until(param_closer)) +
                                         tool_arg_close(literal(param_closer)));
                arg_choice |= arg_rule;
            }
            args = zero_or_more(arg_choice + space());
        }

        // Build function parser: <function=name>args</function>
        auto tool_parser = tool(tool_open(literal(func_opener) + tool_name(literal(name)) + literal(func_name_suffix)) +
                                space() + tool_args(args) + space() + tool_close(literal(func_closer)));

        tool_choices |= rule("tool-" + name, tool_parser);
    }

    // Build the section with markers
    auto section =
        parallel_tool_calls ?
            trigger_rule("tool-call", literal(section_start) + space() + one_or_more(tool_choices + space()) +
                                          literal(section_end)) :
            trigger_rule("tool-call", literal(section_start) + space() + tool_choices + space() + literal(section_end));

    return force_tool_calls ? section : optional(section);
}

// Like python_value(), but the leaf also accepts JSON-cased true/false/null, used by LFM2/LFM2.5
common_peg_parser common_chat_peg_builder::python_or_json_value() {
    return rule("python-or-json-value", [this]() {
        auto ws    = space();
        auto value = python_or_json_value();

        auto member  = sequence({ python_string(), ws, literal(":"), ws, value });
        auto members = sequence({ member, zero_or_more(sequence({ ws, literal(","), ws, member })) });
        auto dict    = rule("python-or-json-dict", [&]() {
            return sequence({ literal("{"), ws, choice({ literal("}"), sequence({ members, ws, literal("}") }) }), ws });
        });

        auto elements = sequence({ value, zero_or_more(sequence({ literal(","), ws, value })) });
        auto array    = rule("python-or-json-array", [&]() {
            return sequence({ literal("["), ws, choice({ literal("]"), sequence({ elements, ws, literal("]") }) }), ws });
        });

        return choice({ dict, array, python_string(), python_number(),
                        python_bool(), python_null(), json_bool(), json_null() });
    });
}

// Python-style tool calls: name(arg1="value1", arg2=123)
// Used only by LFM2 for now, so we don't merge it into autoparser
common_peg_parser common_chat_peg_builder::python_style_tool_calls(
    const ordered_json & tools,
    bool                 parallel_tool_calls,
    bool                 allow_json_literals) {
    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = common_chat_tool_parameters(function);

        auto args = eps();
        if (params.contains("properties") && !params["properties"].empty()) {
            auto arg_choice = choice();
            for (const auto & el : params["properties"].items()) {
                const std::string & prop_name = el.key();
                const auto & prop_def = el.value();
                bool is_string_type = (prop_def.contains("type") && prop_def["type"] == "string");

                auto arg_name_parser = literal(prop_name);

                common_peg_parser arg_value_parser = eps();
                // Quoted literal as a value: normalize_quotes_to_json preserves escapes.
                auto string_value_parser = tool_arg_value(choice({
                    literal("\"") + string_content('"') + literal("\""),
                    literal("'") + string_content('\'') + literal("'")
                }));

                if (is_string_type) {
                    arg_value_parser = string_value_parser;
                } else {
                    arg_value_parser = tool_arg_value(allow_json_literals ? python_or_json_value() : python_value());
                }

                // Full argument: name="value" or name=value
                auto arg_rule = tool_arg(
                    tool_arg_open(tool_arg_name(arg_name_parser) + literal("=")) +
                    arg_value_parser +
                    tool_arg_close(eps())
                );
                arg_choice |= arg_rule;
            }

            args = arg_choice + zero_or_more("," + space() + arg_choice);
        }

        auto tool_parser = tool(tool_open(tool_name(literal(name)) + literal("(")) +
            space() + tool_args(args) + space() + tool_close(literal(")"))
        );

        tool_choices |= rule("tool-" + name, tool_parser);
    }

    if (parallel_tool_calls) {
        return "[" + space() + tool_choices + zero_or_more("," + space() + tool_choices) + space() + "]";
    }
    return "[" + space() + tool_choices + space() + "]";
}

// Helper: Parse dot notation key into prefix and field name
static std::pair<std::string, std::string> parse_key_spec(const std::string & key) {
    auto dot_pos = key.find('.');
    if (dot_pos == std::string::npos) {
        return {"", key};  // Top-level field
    }
    return {key.substr(0, dot_pos), key.substr(dot_pos + 1)};
}

// Mode 1: function_is_key — parse {"function_name": {...}}
common_peg_parser common_chat_peg_builder::build_json_tools_function_is_key(
    const ordered_json & tools,
    const std::string &  args_key,
    const std::string &  effective_args_key,
    const std::string &  call_id_key,
    const std::string &  gen_call_id_key) {

    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = common_chat_tool_parameters(function);

        // Build inner object fields
        std::vector<common_peg_parser> inner_fields;

        if (!call_id_key.empty()) {
            auto id_parser = atomic(
                literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                literal("\"") + tool_id(string_content('"')) + literal("\"")
            );
            inner_fields.push_back(optional(id_parser + space() + optional(literal(",") + space())));
        }

        if (!gen_call_id_key.empty()) {
            auto gen_id_parser = atomic(
                literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(string_content('"')) + literal("\""),
                    tool_id(json_number())
                })
            );
            inner_fields.push_back(optional(gen_id_parser + space() + optional(literal(",") + space())));
        }

        // Arguments — either wrapped in args_key or parsed directly
        common_peg_parser args_parser = eps();
        if (args_key.empty()) {
            args_parser = tool_args(schema(json(), "tool-" + name + "-schema", params));
        } else {
            args_parser = literal("\"" + effective_args_key + "\"") + space() + literal(":") + space() +
                          tool_args(schema(json(), "tool-" + name + "-schema", params));
        }
        inner_fields.push_back(args_parser);

        // Build inner object parser
        common_peg_parser inner_object = eps();
        if (args_key.empty() && inner_fields.size() == 1) {
            inner_object = inner_fields[0];
        } else {
            inner_object = literal("{") + space();
            for (size_t i = 0; i < inner_fields.size(); i++) {
                inner_object = inner_object + inner_fields[i];
                if (i < inner_fields.size() - 1) {
                    inner_object = inner_object + space();
                }
            }
            inner_object = inner_object + space() + literal("}");
        }

        auto tool_parser = tool(
            tool_open(literal("{")) + space() +
            literal("\"") + tool_name(literal(name)) + literal("\"") +
            space() + literal(":") + space() +
            inner_object +
            space() + tool_close(literal("}"))
        );

        tool_choices |= rule("tool-" + name, tool_parser);
    }

    return tool_choices;
}

// Mode 2: Nested keys (dot notation like "function.name")
common_peg_parser common_chat_peg_builder::build_json_tools_nested_keys(
    const ordered_json & tools,
    const std::string &  effective_name_key,
    const std::string &  effective_args_key,
    const std::string &  call_id_key,
    const std::string &  gen_call_id_key) {

    auto tool_choices = choice();

    auto name_spec = parse_key_spec(effective_name_key);
    auto args_spec = parse_key_spec(effective_args_key);

    std::string nested_prefix     = !name_spec.first.empty() ? name_spec.first  : args_spec.first;
    std::string nested_name_field = !name_spec.first.empty() ? name_spec.second  : effective_name_key;
    std::string nested_args_field = !args_spec.first.empty() ? args_spec.second  : effective_args_key;

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = common_chat_tool_parameters(function);

        auto nested_name = literal("\"" + nested_name_field + "\"") + space() + literal(":") + space() +
                          atomic(literal("\"") + tool_name(literal(name)) + literal("\""));
        auto nested_args = literal("\"" + nested_args_field + "\"") + space() + literal(":") + space() +
                          tool_args(schema(json(), "tool-" + name + "-schema", params));

        auto nested_object = literal("{") + space() +
                            nested_name + space() + literal(",") + space() +
                            nested_args +
                            space() + literal("}");

        // Format: { id?, "function": {...} }
        auto tool_parser_body = tool_open(literal("{")) + space();

        if (!call_id_key.empty()) {
            auto id_spec = parse_key_spec(call_id_key);
            if (id_spec.first.empty()) {
                auto id_parser = atomic(
                    literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                    literal("\"") + tool_id(string_content('"')) + literal("\"")
                );
                tool_parser_body = tool_parser_body + optional(id_parser + space() + literal(",") + space());
            }
        }

        if (!gen_call_id_key.empty()) {
            auto gen_id_spec = parse_key_spec(gen_call_id_key);
            if (gen_id_spec.first.empty()) {
                auto gen_id_parser = atomic(
                    literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                    choice({
                        literal("\"") + tool_id(string_content('"')) + literal("\""),
                        tool_id(json_number())
                    })
                );
                tool_parser_body = tool_parser_body + optional(gen_id_parser + space() + literal(",") + space());
            }
        }

        auto nested_field = literal("\"" + nested_prefix + "\"") + space() + literal(":") + space() + nested_object;
        tool_parser_body = tool_parser_body + nested_field + space() + tool_close(literal("}"));

        tool_choices |= rule("tool-" + name, tool(tool_parser_body));
    }

    return tool_choices;
}

// Mode 3: Flat keys with optional ID fields and parameter ordering
common_peg_parser common_chat_peg_builder::build_json_tools_flat_keys(
    const ordered_json &             tools,
    const std::string &              effective_name_key,
    const std::string &              effective_args_key,
    const std::string &              call_id_key,
    const std::string &              gen_call_id_key,
    const std::vector<std::string> & parameters_order,
    bool                             accept_openai_wrapper) {

    auto tool_choices    = choice();
    auto name_key_parser = literal("\"" + effective_name_key + "\"");
    auto args_key_parser = literal("\"" + effective_args_key + "\"");

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = common_chat_tool_parameters(function);

        auto tool_name_ = name_key_parser + space() + literal(":") + space() +
                         atomic(literal("\"") + tool_name(literal(name)) + literal("\""));
        auto tool_args_ = args_key_parser + space() + literal(":") + space() +
                         tool_args(schema(json(), "tool-" + name + "-schema", params));

        // Build ID parsers if keys are provided
        common_peg_parser id_parser = eps();
        if (!call_id_key.empty()) {
            id_parser = atomic(
                literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(string_content('"')) + literal("\""),
                    tool_id(json_number())
                })
            );
        }

        common_peg_parser gen_id_parser = eps();
        if (!gen_call_id_key.empty()) {
            gen_id_parser = atomic(
                literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(string_content('"')) + literal("\""),
                    tool_id(json_number())
                })
            );
        }

        // Create (parser, key) pairs for all fields, then sort by parameters_order
        std::vector<std::pair<common_peg_parser, std::string>> parser_pairs;
        parser_pairs.emplace_back(tool_name_, effective_name_key);
        parser_pairs.emplace_back(tool_args_, effective_args_key);
        if (!call_id_key.empty()) {
            parser_pairs.emplace_back(optional(id_parser), call_id_key);
        }
        if (!gen_call_id_key.empty()) {
            parser_pairs.emplace_back(optional(gen_id_parser), gen_call_id_key);
        }

        std::sort(parser_pairs.begin(), parser_pairs.end(),
            [&parameters_order](const auto & a, const auto & b) {
                auto pos_a = std::find(parameters_order.begin(), parameters_order.end(), a.second);
                auto pos_b = std::find(parameters_order.begin(), parameters_order.end(), b.second);
                size_t idx_a = (pos_a == parameters_order.end()) ? parameters_order.size() : std::distance(parameters_order.begin(), pos_a);
                size_t idx_b = (pos_b == parameters_order.end()) ? parameters_order.size() : std::distance(parameters_order.begin(), pos_b);
                return idx_a < idx_b;
            });

        // accept an optional leading "type": "function" field when the model emits the OpenAI wrapper
        common_peg_parser type_field = eps();
        if (accept_openai_wrapper) {
            type_field = optional(literal("\"type\"") + space() + literal(":") + space() +
                                  literal("\"function\"") + space() + literal(",") + space());
        }
        auto ordered_body = tool_open(literal("{")) + space() + type_field;
        for (size_t i = 0; i < parser_pairs.size(); i++) {
            ordered_body = ordered_body + parser_pairs[i].first;
            if (i < parser_pairs.size() - 1) {
                ordered_body = ordered_body + space() + literal(",") + space();
            }
        }
        ordered_body = ordered_body + space() + tool_close(literal("}"));

        tool_choices |= rule("tool-" + name, tool(ordered_body));
    }

    return tool_choices;
}

common_peg_parser common_chat_peg_builder::prefix(const std::string & s, const std::string & delimiter) {
    if (s.empty()) {
        return eps();
    }
    if (delimiter.empty()) {
        return literal(s);
    }
    return literal(s.substr(0, s.find(delimiter)));
}

common_peg_parser common_chat_peg_builder::optspace(const std::string & tag) {
    auto parser = eps();
    size_t end_of_prefix_space = tag.size();
    size_t start_of_suffix_space = tag.size();
    for (size_t i = 0; i < tag.size(); i++) {
        if (!std::isspace(tag[i])) {
            end_of_prefix_space = i;
            break;
        }
    }
    for (size_t i = tag.size(); i > 0; i--) {
        if (!std::isspace(tag[i - 1])) {
            start_of_suffix_space = i;
            break;
        }
    }
    for (size_t i = 0; i < end_of_prefix_space; i++) {
        parser += optional(literal(std::string(1, tag[i])));
    }
    parser += literal(tag.substr(end_of_prefix_space, start_of_suffix_space - end_of_prefix_space));
    for (size_t i = start_of_suffix_space; i < tag.size(); i++) {
        parser += optional(literal(std::string(1, tag[i])));
    }
    return parser;
}

common_peg_parser common_chat_peg_builder::standard_json_tools(
                                                       const std::string &              section_start,
                                                       const std::string &              section_end,
                                                       const ordered_json &             tools,
                                                       bool                             parallel_tool_calls,
                                                       bool                             force_tool_calls,
                                                       const std::string &              name_key,
                                                       const std::string &              args_key,
                                                       bool                             array_wrapped,
                                                       bool                             function_is_key,
                                                       const std::string &              call_id_key,
                                                       const std::string &              gen_call_id_key,
                                                       const std::vector<std::string> & parameters_order,
                                                       bool                             accept_openai_wrapper) {
    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    std::string effective_name_key = name_key.empty() ? "name" : name_key;
    std::string effective_args_key = args_key.empty() ? "arguments" : args_key;

    // Dispatch to the appropriate builder based on the JSON layout mode
    common_peg_parser tool_choices = eps();
    if (function_is_key) {
        tool_choices = build_json_tools_function_is_key(tools, args_key, effective_args_key, call_id_key, gen_call_id_key);
    } else {
        auto name_spec = parse_key_spec(effective_name_key);
        auto args_spec = parse_key_spec(effective_args_key);
        if (!name_spec.first.empty() || !args_spec.first.empty()) {
            tool_choices = build_json_tools_nested_keys(tools, effective_name_key, effective_args_key, call_id_key, gen_call_id_key);
        } else {
            tool_choices = build_json_tools_flat_keys(tools, effective_name_key, effective_args_key, call_id_key, gen_call_id_key, parameters_order, accept_openai_wrapper);
        }
    }

    // Build the section with markers
    auto tool_calls = tool_choices;
    if (parallel_tool_calls) {
        tool_calls = tool_calls + zero_or_more(space() + literal(",") + space() + tool_choices);
    }

    if (array_wrapped) {
        tool_calls = literal("[") + space() + tool_calls + space() + literal("]");
    }

    auto section =
        trigger_rule("tool-call", literal(section_start) + space() + tool_calls + space() + literal(section_end));

    return force_tool_calls ? section : optional(section);
}

void common_chat_peg_gemma4_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    for (const auto & node : result.nodes) {
        visit(arena, node);
    }
}

static std::string gemma4_to_json(const common_peg_ast_arena & arena, common_peg_ast_id id) {
    const auto & node = arena.get(id);

    if (node.text.empty()) {
        return "";
    }

    if (node.rule == "gemma4-number" || node.rule == "gemma4-bool" || node.rule == "gemma4-null") {
        return std::string(node.text);
    }

    if (node.rule == "gemma4-string-content") {
        return escape_json_string_inner(std::string(node.text));
    }

    if (node.rule == "gemma4-string") {
        std::string result = "\"";
        if (!node.children.empty()) {
            result += gemma4_to_json(arena, node.children[0]);
            if (!node.is_partial) {
                result += "\"";
            }
        }
        return result;
    }

    if (node.rule == "gemma4-array") {
        std::string result = "[";

        bool add_comma = false;
        for (auto child_id : node.children) {
            if (add_comma) {
                result += ',';
            }
            add_comma = true;
            result += gemma4_to_json(arena, child_id);
        }

        if (!node.is_partial) {
            result += ']';
        }
        return result;
    }

    if (node.rule == "gemma4-dict-key-name") {
        return std::string(node.text);
    }

    if (node.rule == "gemma4-dict-key") {
        std::string result = "\"";
        if (!node.children.empty()) {
            result += escape_json_string_inner(gemma4_to_json(arena, node.children[0]));
        }
        if (!node.is_partial) {
            result += "\":";
        }
        return result;
    }

    if (node.rule == "gemma4-dict-kv") {
        std::string result;
        for (auto child_id : node.children) {
            result += gemma4_to_json(arena, child_id);
        }
        return result;
    }

    if (node.rule == "gemma4-dict") {
        std::string result = "{";

        bool add_comma = false;
        for (auto child_id : node.children) {
            if (add_comma) {
                result += ',';
            }
            add_comma = true;
            result += gemma4_to_json(arena, child_id);
        }

        if (!node.is_partial) {
            result += '}';
        }
        return result;
    }

    if (node.rule == "gemma4-value") {
        if (!node.children.empty()) {
            return gemma4_to_json(arena, node.children[0]);
        }
        return "";
    }

    return "";
}

void common_chat_peg_gemma4_mapper::visit(const common_peg_ast_arena & arena, common_peg_ast_id id) {
    const auto & node = arena.get(id);

    if (node.tag == "reasoning") {
        result.reasoning_content += node.sanitized_text();
        return;
    }

    if (node.tag == "content") {
        result.content += node.sanitized_text();
        return;
    }

    if (node.tag == "tool") {
        auto name_id = arena.find_by_tag(node, "tool-name");
        auto args_id = arena.find_by_tag(node, "tool-args");

        if (name_id != COMMON_PEG_INVALID_AST_ID && args_id != COMMON_PEG_INVALID_AST_ID) {
            const auto & name_node = arena.get(name_id);
            const auto & args_node = arena.get(args_id);

            if (!name_node.is_partial) {
                common_chat_tool_call call;
                call.name = std::string(name_node.text);
                if (!args_node.children.empty()) {
                    call.arguments = gemma4_to_json(arena, args_node.children[0]);
                }
                result.tool_calls.push_back(call);
            }
        }

        return;
    }

    for (auto child_id : node.children) {
        visit(arena, child_id);
    }
}

static void minimax_m3_collect(const common_peg_ast_arena &     arena,
                               const common_peg_ast_node &      node,
                               const std::string &              tag,
                               std::vector<common_peg_ast_id> & out) {
    for (auto child_id : node.children) {
        const auto & child = arena.get(child_id);
        if (child.tag == tag) {
            out.push_back(child_id);
        } else {
            minimax_m3_collect(arena, child, tag, out);
        }
    }
}

static common_peg_ast_id minimax_m3_value_of(const common_peg_ast_arena & arena, const common_peg_ast_node & node) {
    for (auto child_id : node.children) {
        const auto & tag = arena.get(child_id).tag;
        if (tag == common_chat_peg_builder::TOOL_ARG_VALUE ||
            tag == common_chat_peg_builder::TOOL_ARG_STRING_VALUE ||
            tag == common_chat_peg_minimax_m3_mapper::TOOL_ARG_OBJECT ||
            tag == common_chat_peg_minimax_m3_mapper::TOOL_ARG_ARRAY) {
            return child_id;
        }
    }
    return COMMON_PEG_INVALID_AST_ID;
}

static std::string minimax_m3_value_to_json(const common_peg_ast_arena & arena, common_peg_ast_id id, bool closed);

static std::string minimax_m3_member_to_json(const common_peg_ast_arena & arena, const common_peg_ast_node & node) {
    auto name_id = arena.find_by_tag(node, common_chat_peg_builder::TOOL_ARG_NAME);
    if (name_id == COMMON_PEG_INVALID_AST_ID) {
        return "";
    }

    return ordered_json(arena.get(name_id).text).dump() + ":" +
           minimax_m3_value_to_json(arena, minimax_m3_value_of(arena, node), !node.is_partial);
}

static std::string minimax_m3_container_to_json(const common_peg_ast_arena & arena,
                                                const common_peg_ast_node & node,
                                                bool                        is_object,
                                                bool                        closed) {
    const std::string tag = is_object ? common_chat_peg_builder::TOOL_ARG
                                      : common_chat_peg_minimax_m3_mapper::TOOL_ARG_ITEM;

    std::vector<common_peg_ast_id> entries;
    minimax_m3_collect(arena, node, tag, entries);

    std::string result = is_object ? "{" : "[";

    bool add_comma = false;
    for (auto entry_id : entries) {
        const auto & entry = arena.get(entry_id);

        std::string text;
        if (is_object) {
            text = minimax_m3_member_to_json(arena, entry);
        } else {
            text = minimax_m3_value_to_json(arena, minimax_m3_value_of(arena, entry), !entry.is_partial);
        }

        if (text.empty()) {
            continue;
        }

        if (add_comma) {
            result += ",";
        }
        add_comma = true;
        result += text;
    }

    if (closed) {
        result += is_object ? "}" : "]";
    }
    return result;
}

static std::string minimax_m3_value_to_json(const common_peg_ast_arena & arena, common_peg_ast_id id, bool closed) {
    if (id == COMMON_PEG_INVALID_AST_ID) {
        return "";
    }

    const auto & node = arena.get(id);

    if (node.tag == common_chat_peg_minimax_m3_mapper::TOOL_ARG_OBJECT) {
        return minimax_m3_container_to_json(arena, node, /* is_object = */ true, closed);
    }

    if (node.tag == common_chat_peg_minimax_m3_mapper::TOOL_ARG_ARRAY) {
        return minimax_m3_container_to_json(arena, node, /* is_object = */ false, closed);
    }

    if (node.tag == common_chat_peg_builder::TOOL_ARG_STRING_VALUE) {
        return "\"" + escape_json_string_inner(std::string(node.text)) + (closed ? "\"" : "");
    }

    // Numbers and booleans are written verbatim by the template
    return std::string(node.text);
}

void common_chat_peg_minimax_m3_mapper::from_ast(const common_peg_ast_arena &    arena,
                                                 const common_peg_parse_result & result) {
    for (const auto & node : result.nodes) {
        visit(arena, node);
    }
}

void common_chat_peg_minimax_m3_mapper::visit(const common_peg_ast_arena & arena, common_peg_ast_id id) {
    const auto & node = arena.get(id);

    if (node.tag == common_chat_peg_builder::REASONING) {
        result.reasoning_content += node.sanitized_text();
        return;
    }

    if (node.tag == common_chat_peg_builder::CONTENT) {
        result.content += node.sanitized_text();
        return;
    }

    if (node.tag == common_chat_peg_builder::TOOL) {
        auto name_id = arena.find_by_tag(node, common_chat_peg_builder::TOOL_NAME);
        if (name_id != COMMON_PEG_INVALID_AST_ID) {
            common_chat_tool_call call;
            call.name      = std::string(arena.get(name_id).text);
            call.arguments = minimax_m3_container_to_json(arena, node, /* is_object = */ true, !node.is_partial);
            result.tool_calls.push_back(call);
        }
        return;
    }

    for (auto child_id : node.children) {
        visit(arena, child_id);
    }
}

// file: common/chat-peg-parser.h
#pragma once

#include "chat.h"
#include "peg-parser.h"

#include <map>
#include <optional>
#include <vector>

class common_chat_peg_mapper {
  public:
    common_chat_msg & result;

    common_chat_peg_mapper(common_chat_msg & msg) : result(msg) {}

    virtual ~common_chat_peg_mapper() = default;

    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
    virtual void map(const common_peg_ast_node & node);
  protected:
    virtual std::string normalize_container_value(const std::string & input);
  private:
      // Tool call handling state
      std::optional<common_chat_tool_call> pending_tool_call;  // Tool call waiting for name
      common_chat_tool_call *              current_tool          = nullptr;
      int                                  arg_count             = 0;
      bool                                 closing_quote_pending = false;
      std::string                          args_buffer;  // Buffer to delay arguments until tool name is known

      // Returns a reference to the active argument destination string.
      // Before tool_name is known, writes go to args_buffer; after, to current_tool->arguments.
      std::string & args_target();
};

class common_chat_peg_gemma4_mapper : public common_chat_peg_mapper {
  public:
    common_chat_peg_gemma4_mapper(common_chat_msg & msg) : common_chat_peg_mapper(msg) {}
    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
  private:
    void visit(const common_peg_ast_arena & arena, common_peg_ast_id id);
};

class common_chat_peg_minimax_m3_mapper : public common_chat_peg_mapper {
  public:
    static constexpr const char * TOOL_ARG_OBJECT = "tool-arg-object";
    static constexpr const char * TOOL_ARG_ARRAY  = "tool-arg-array";
    static constexpr const char * TOOL_ARG_ITEM   = "tool-arg-item";

    common_chat_peg_minimax_m3_mapper(common_chat_msg & msg) : common_chat_peg_mapper(msg) {}
    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
  private:
    void visit(const common_peg_ast_arena & arena, common_peg_ast_id id);
};

struct content_structure;
struct tool_call_structure;

constexpr size_t COMMON_CHAT_MAX_PERMUTE = 6;

class common_chat_peg_builder : public common_peg_parser_builder {
  public:
    // Tag constants (from former common_chat_peg_base_builder)
    static constexpr const char * REASONING_BLOCK = "reasoning-block";
    static constexpr const char * REASONING       = "reasoning";
    static constexpr const char * CONTENT         = "content";

    // Tag constants
    static constexpr const char * TOOL           = "tool";
    static constexpr const char * TOOL_OPEN      = "tool-open";
    static constexpr const char * TOOL_CLOSE     = "tool-close";
    static constexpr const char * TOOL_ID        = "tool-id";
    static constexpr const char * TOOL_NAME      = "tool-name";
    static constexpr const char * TOOL_ARGS      = "tool-args";
    static constexpr const char * TOOL_ARG       = "tool-arg";
    static constexpr const char * TOOL_ARG_OPEN  = "tool-arg-open";
    static constexpr const char * TOOL_ARG_CLOSE = "tool-arg-close";
    static constexpr const char * TOOL_ARG_NAME         = "tool-arg-name";
    static constexpr const char * TOOL_ARG_VALUE        = "tool-arg-value";
    static constexpr const char * TOOL_ARG_STRING_VALUE = "tool-arg-string-value";  // For schema-declared string types

    // Low-level tag methods (from former common_chat_peg_base_builder)
    common_peg_parser reasoning_block(const common_peg_parser & p) { return tag(REASONING_BLOCK, p); }

    common_peg_parser reasoning(const common_peg_parser & p) { return tag(REASONING, p); }

    common_peg_parser content(const common_peg_parser & p) { return tag(CONTENT, p); }

    common_peg_parser tag_with_safe_content(const std::string &       tag_name,
                        const std::string &       marker,
                        const common_peg_parser & p);

    // Low-level tag methods
    common_peg_parser tool(const common_peg_parser & p) { return tag(TOOL, p); }
    common_peg_parser tool_open(const common_peg_parser & p) { return atomic(tag(TOOL_OPEN, p)); }
    common_peg_parser tool_close(const common_peg_parser & p) { return atomic(tag(TOOL_CLOSE, p)); }
    common_peg_parser tool_id(const common_peg_parser & p) { return atomic(tag(TOOL_ID, p)); }
    common_peg_parser tool_name(const common_peg_parser & p) { return atomic(tag(TOOL_NAME, p)); }
    common_peg_parser tool_args(const common_peg_parser & p) { return tag(TOOL_ARGS, p); }
    common_peg_parser tool_arg(const common_peg_parser & p) { return tag(TOOL_ARG, p); }
    common_peg_parser tool_arg_open(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_OPEN, p)); }
    common_peg_parser tool_arg_close(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_CLOSE, p)); }
    common_peg_parser tool_arg_name(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_NAME, p)); }
    common_peg_parser tool_arg_value(const common_peg_parser & p) { return tag(TOOL_ARG_VALUE, p); }

    // Use for schema-declared string types - won't be treated as potential JSON container
    common_peg_parser tool_arg_string_value(const common_peg_parser & p) { return tag(TOOL_ARG_STRING_VALUE, p); }
    common_peg_parser tool_arg_json_value(const common_peg_parser & p) { return tag(TOOL_ARG_VALUE, p); }


    // Matches every parser exactly once, in any order.
    common_peg_parser permute(const std::string & rule_prefix, const std::vector<common_peg_parser> & parsers);

    // Return a parser that parses the prefix of a string, up to a given delimiter.
    common_peg_parser prefix(const std::string & s, const std::string & delimiter = {});

    // Return a parser that parses all elements of tag, but leading and trailing spaces are optional
    common_peg_parser optspace(const std::string & tag);

    // Legacy-compatible helper for building standard JSON tool calls
    // Used by tests and manual parsers
    // name_key/args_key: JSON key names for function name and arguments
    //   Empty or "name"/"arguments" will accept both common variations
    //   Supports dot notation for nested objects (e.g., "function.name")
    // array_wrapped: if true, tool calls are wrapped in JSON array [...]
    // function_is_key: if true, function name is the JSON key (e.g., {"func_name": {...}})
    // call_id_key: JSON key for string call ID (e.g., "id")
    // gen_call_id_key: JSON key for generated integer call ID (e.g., "tool_call_id")
    // parameters_order: order in which JSON fields should be parsed
    common_peg_parser standard_json_tools(const std::string &              section_start,
                                          const std::string &              section_end,
                                          const common_json &   tools,
                                          bool                             parallel_tool_calls,
                                          bool                             force_tool_calls,
                                          const std::string &              name_key = "",
                                          const std::string &              args_key = "",
                                          bool                             array_wrapped = false,
                                          bool                             function_is_key = false,
                                          const std::string &              call_id_key = "",
                                          const std::string &              gen_call_id_key = "",
                                          const std::vector<std::string> & parameters_order = {},
                                          bool                             accept_openai_wrapper = false);

    // Legacy-compatible helper for building XML/tagged style tool calls
    // Used by tests and manual parsers
    common_peg_parser standard_constructed_tools(const std::map<std::string, std::string> & markers,
                                                 const common_json &             tools,
                                                 bool                                       parallel_tool_calls,
                                                 bool                                       force_tool_calls);

    // Helper for Python-style function call format: name(arg1="value1", arg2=123)
    // Used by LFM2 and similar templates
    common_peg_parser python_style_tool_calls(const common_json & tools,
                                              bool                           parallel_tool_calls,
                                              bool                           allow_json_literals);

  private:
    // Python values plus JSON true/false/null.
    common_peg_parser python_or_json_value();

    // Implementation helpers for standard_json_tools — one per JSON tool call layout mode
    common_peg_parser build_json_tools_function_is_key(const common_json & tools,
                                                       const std::string &            args_key,
                                                       const std::string &            effective_args_key,
                                                       const std::string &            call_id_key,
                                                       const std::string &            gen_call_id_key);

    common_peg_parser build_json_tools_nested_keys(const common_json & tools,
                                                   const std::string &            effective_name_key,
                                                   const std::string &            effective_args_key,
                                                   const std::string &            call_id_key,
                                                   const std::string &            gen_call_id_key);

    common_peg_parser build_json_tools_flat_keys(const common_json &   tools,
                                                 const std::string &              effective_name_key,
                                                 const std::string &              effective_args_key,
                                                 const std::string &              call_id_key,
                                                 const std::string &              gen_call_id_key,
                                                 const std::vector<std::string> & parameters_order,
                                                 bool                             accept_openai_wrapper);
};

inline common_peg_arena build_chat_peg_parser(
  const std::function<common_peg_parser(common_chat_peg_builder & builder)> & fn) {
  common_chat_peg_builder builder;
  builder.set_root(fn(builder));
  return builder.build();
}

class tag_based_peg_mapper {
  public:
    std::map<std::string, std::string> tags;

    void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
};

struct tagged_parse_result {
    common_peg_parse_result              result;
    std::map<std::string, std::string> tags;
};

struct tagged_peg_parser {
    common_peg_arena arena;
    common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_NONE;

    tagged_peg_parser & withDebug() {
      flags |= COMMON_PEG_PARSE_FLAG_DEBUG;
      return *this;
    }

    tagged_peg_parser & withoutDebug() {
      flags = flags & ~COMMON_PEG_PARSE_FLAG_DEBUG;
      return *this;
    }

    tagged_parse_result parse_and_extract(const std::string & input, common_peg_parse_flags extra_flags = COMMON_PEG_PARSE_FLAG_NONE) const;
    tagged_parse_result parse_anywhere_and_extract(const std::string & input) const;
};

tagged_peg_parser build_tagged_peg_parser(
    const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn);

// file: common/chat.cpp
#include "chat.h"

#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "ggml.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "log.h"
#include "parsers/parsers.h"

#include "jinja/value.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"
#include "peg-parser.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <iomanip>
#include <map>

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = common_json;

static std::string format_time(const std::chrono::system_clock::time_point & now, const std::string & format) {
    auto               time       = std::chrono::system_clock::to_time_t(now);
    auto               local_time = *std::localtime(&time);
    std::ostringstream ss;
    ss << std::put_time(&local_time, format.c_str());
    auto res = ss.str();
    return res;
}

static json safe_args_parse(const std::string & to_parse) {
    std::string stripped = to_parse;
    if (to_parse.at(0) == '"' && to_parse.at(to_parse.length() - 1) == '"') {
        stripped = to_parse.substr(1, to_parse.length() - 1);
    }
    try {
        return json::parse(stripped);
    } catch (const common_json_error & e) {
        return stripped;
    }
}

static std::string string_diff(const std::string & last, const std::string & current) {
    if (last.empty()) {
        return current;
    }
    if (!string_starts_with(current, last)) {
        if (string_starts_with(last, current)) {
            // This happens if the last generation ended on a partial stop word (not erased),
            // and the current ended on a stop word (erased).
            return "";
        }
        throw std::runtime_error("Invalid diff: '" + last + "' not found at start of '" + current + "'");
    }
    return current.substr(last.size());
}

static bool has_content_or_tool_calls(const common_chat_msg & msg) {
    return !msg.content.empty() || !msg.tool_calls.empty();
}

std::string common_chat_msg::render_content(const std::string & delimiter) const {
    if (!content.empty() && !content_parts.empty()) {
        throw std::runtime_error("Cannot specify both content and content_parts");
    }
    if (!content.empty()) {
        return content;
    }

    std::string text;
    for (const auto & part : content_parts) {
        if (part.type == "text") {
            if (!text.empty()) {
                text += delimiter;
            }
            text += part.text;
        }
    }
    return text;
}

common_chat_role common_chat_role_from_string(const std::string & role) {
    if (role == "system")    { return COMMON_CHAT_ROLE_SYSTEM;    }
    if (role == "assistant") { return COMMON_CHAT_ROLE_ASSISTANT; }
    if (role == "user")      { return COMMON_CHAT_ROLE_USER;      }
    if (role == "tool")      { return COMMON_CHAT_ROLE_TOOL;      }
    return COMMON_CHAT_ROLE_UNKNOWN;
}

const char * common_chat_role_to_string(common_chat_role role) {
    switch (role) {
        case COMMON_CHAT_ROLE_SYSTEM:    return "system";
        case COMMON_CHAT_ROLE_ASSISTANT: return "assistant";
        case COMMON_CHAT_ROLE_USER:      return "user";
        case COMMON_CHAT_ROLE_TOOL:      return "tool";
        case COMMON_CHAT_ROLE_UNKNOWN:   return "";
    }
    return "";
}

json common_chat_msg_delimiters::to_json() const {
    json result = json::array();
    for (const auto & d : delimiters) {
        result.push_back({
            { "role",      common_chat_role_to_string(d.role) },
            { "delimiter", d.delimiter                        },
        });
    }
    return result;
}

common_chat_msg_delimiters common_chat_msg_delimiters_parse(const json & delimiters) {
    common_chat_msg_delimiters result;

    if (!delimiters.is_array()) {
        return result;
    }

    result.delimiters.reserve(delimiters.size());
    for (const auto & d : delimiters) {
        if (!d.is_object()) {
            continue;
        }
        result.delimiters.push_back({
            common_chat_role_from_string(d.value("role", std::string())),
            d.value("delimiter", std::string()),
        });
    }

    return result;
}

void common_chat_msg_delimiters::tokenize(const llama_vocab * vocab) {
    for (auto & d : delimiters) {
        d.tokens = common_tokenize(vocab, d.delimiter, false, true);
    }
}

common_chat_msg_spans common_chat_msg_delimiters::split(const llama_tokens & tokens, const std::map<size_t, size_t> & skips) const {
    std::vector<std::pair<common_chat_role, size_t>> matches;

    auto skip = skips.begin();
    for (size_t i = 0; i < tokens.size();) {
        if (skip != skips.end() && i == skip->first) {
            i += skip->second;
            ++skip;
            continue;
        }
        for (const auto & d : delimiters) {
            if (i + d.tokens.size() > tokens.size()) {
                continue;
            }
            if (std::equal(d.tokens.begin(), d.tokens.end(), tokens.begin() + i)) {
                matches.emplace_back(d.role, i);
                break;
            }
        }
        i++;
    }

    matches.emplace_back(COMMON_CHAT_ROLE_UNKNOWN, tokens.size());

    common_chat_msg_spans spans;
    for (size_t i = 0; i + 1 < matches.size(); i++) {
        const auto & curr = matches[i];
        const auto & next = matches[i + 1];
        spans.add(curr.first, curr.second, next.second - curr.second);
    }

    return spans;
}

json common_chat_msg::to_json_oaicompat(bool concat_typed_text) const {
    if (!content.empty() && !content_parts.empty()) {
        throw std::runtime_error("Cannot specify both content and content_parts");
    }
    json jmsg {
        {"role", role},
    };
    if (!content.empty()) {
        jmsg["content"] = content;
    } else if (!content_parts.empty()) {
        if (concat_typed_text || contains_media()) {
            std::string text;
            bool last_was_media_marker = false;
            // join parts with newline, do not add newline before or after media markers
            for (const auto & part : content_parts) {
                bool add_new_line = true;
                if (part.type == "text") {
                    add_new_line = !last_was_media_marker && !text.empty();
                    last_was_media_marker = false;
                } else if (part.type == "media_marker") {
                    add_new_line = false;
                    last_was_media_marker = true;
                } else {
                    LOG_WRN("Ignoring content part type: %s\n", part.type.c_str());
                    continue;
                }

                if (add_new_line) {
                    text += '\n';
                }

                text += part.text;
            }
            jmsg["content"] = text;
        } else {
            auto & parts = jmsg["content"] = json::array();
            for (const auto & part : content_parts) {
                parts.push_back({
                    {"type", part.type},
                    {"text", part.text},
                });
            }
        }
    } else {
        jmsg["content"] = "";
    }
    if (!reasoning_content.empty()) {
        jmsg["reasoning_content"] = reasoning_content;
    }
    if (!tool_name.empty()) {
        jmsg["name"] = tool_name;
    }
    if (!tool_call_id.empty()) {
        jmsg["tool_call_id"] = tool_call_id;
    }
    if (!tool_calls.empty()) {
        jmsg["tool_calls"] = json::array();
        auto & jtool_calls = jmsg["tool_calls"];
        for (const auto & tool_call : tool_calls) {
            json tc {
                {"type", "function"},
                {"function", {
                    {"name", tool_call.name},
                    {"arguments", json(tool_call.arguments)},
                }},
            };
            if (!tool_call.id.empty()) {
                tc["id"] = tool_call.id;
            }
            // Some templates generate and require an id (sometimes in a very specific format, e.g. Mistral Nemo).
            // We only generate a random id for the ones that don't generate one by themselves
            // (they also won't get to see it as their template likely doesn't use it, so it's all for the client)
            // {"id", tc.id.empty() ? gen_tool_call_id() : tc.id},
            jtool_calls.push_back(tc);
        }
    }

    return jmsg;
}

std::vector<common_chat_msg_diff> common_chat_msg_diff::compute_diffs(const common_chat_msg & msg_prv,
                                                                      const common_chat_msg & msg_new) {
    std::vector<common_chat_msg_diff> diffs;
    if (msg_new.tool_calls.size() > msg_prv.tool_calls.size()) {
        diffs.reserve(msg_new.tool_calls.size() - msg_prv.tool_calls.size() + 3);
    } else {
        diffs.reserve(3);
    }

    // TODO: these can become expensive for long messages - how to optimize?
    if (msg_prv.reasoning_content != msg_new.reasoning_content) {
        auto & diff                  = diffs.emplace_back();
        diff.reasoning_content_delta = string_diff(msg_prv.reasoning_content, msg_new.reasoning_content);
    }
    if (msg_prv.content != msg_new.content) {
        auto & diff        = diffs.emplace_back();
        diff.content_delta = string_diff(msg_prv.content, msg_new.content);
    }

    if (msg_new.tool_calls.size() < msg_prv.tool_calls.size()) {
        std::string err = "Invalid diff: now finding less tool calls!\n";
        err += "  Previous (" + std::to_string(msg_prv.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_prv.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current (" + std::to_string(msg_new.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_new.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current msg text content:\n" + msg_new.content + "\n";
        throw std::runtime_error(err);
    }

    if (!msg_prv.tool_calls.empty()) {
        const auto   idx  = msg_prv.tool_calls.size() - 1;
        const auto & pref = msg_prv.tool_calls[idx];
        const auto & newf = msg_new.tool_calls[idx];
        // Allow tool name to change during incremental parsing:
        // - empty -> non-empty (initial discovery)
        // - prefix -> longer string (name grows as more input is parsed)
        if (pref.name != newf.name && !pref.name.empty() && !newf.name.empty()) {
            // Check if one is a prefix of the other (for incremental parsing where names grow or shrink)
            bool is_prefix = (newf.name.rfind(pref.name, 0) == 0);
            if (!is_prefix) {
                LOG_ERR("Tool call mismatch: prev='%s' new='%s'\n", pref.name.c_str(), newf.name.c_str());
                throw std::runtime_error("Invalid diff: tool call mismatch!");
            }
        }
        const auto args_diff = string_diff(pref.arguments, newf.arguments);
        if (!args_diff.empty() || pref.id != newf.id || pref.name != newf.name) {
            auto & diff          = diffs.emplace_back();
            diff.tool_call_index = idx;
            if (pref.id != newf.id || pref.name != newf.name) {
                diff.tool_call_delta.id   = newf.id;
                diff.tool_call_delta.name = newf.name;
            }
            diff.tool_call_delta.arguments = args_diff;
        }
    }
    for (size_t idx = msg_prv.tool_calls.size(); idx < msg_new.tool_calls.size(); ++idx) {
        auto & diff          = diffs.emplace_back();
        diff.tool_call_index = idx;
        diff.tool_call_delta = msg_new.tool_calls[idx];
    }

    return diffs;
}

using chat_template_caps = jinja::caps;

struct common_chat_templates {
    bool add_bos;
    bool add_eos;
    bool has_explicit_template;  // Model had builtin template or template overridden was specified.
    std::unique_ptr<common_chat_template> template_default;  // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
};

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice) {
    if (tool_choice == "auto") {
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
    if (tool_choice == "none") {
        return COMMON_CHAT_TOOL_CHOICE_NONE;
    }
    if (tool_choice == "required") {
        return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    }
    throw std::invalid_argument("Invalid tool_choice: " + tool_choice);
}

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates) {
    common_chat_templates_inputs inputs;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = "test";
    inputs.messages = { msg };
    inputs.enable_thinking = true;
    inputs.add_generation_prompt = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    auto params = common_chat_templates_apply(chat_templates, inputs);
    return params.supports_thinking;
}

std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const json & messages) {
    std::vector<common_chat_msg> msgs;

    try {
        if (!messages.is_array()) {
            throw std::invalid_argument("Expected 'messages' to be an array, got " + messages.dump());
        }

        for (const auto & message : messages) {
            if (!message.is_object()) {
                throw std::invalid_argument("Expected 'message' to be an object, got " + message.dump());
            }

            common_chat_msg msg;
            if (!message.contains("role")) {
                throw std::invalid_argument("Missing 'role' in message: " + message.dump());
            }
            msg.role = message.at("role");

            auto has_content    = message.contains("content");
            auto has_tool_calls = message.contains("tool_calls");
            if (has_content) {
                const auto & content = message.at("content");
                if (content.is_string()) {
                    msg.content = content;
                } else if (content.is_array()) {
                    for (const auto & part : content) {
                        if (!part.contains("type")) {
                            throw std::invalid_argument("Missing content part type: " + part.dump());
                        }
                        const auto & type = part.at("type");
                        if (type != "text" && type != "media_marker") {
                            throw std::invalid_argument("Unsupported content part type: " + type.dump());
                        }
                        common_chat_msg_content_part msg_part;
                        msg_part.type = type;
                        msg_part.text = part.at("text");
                        msg.content_parts.push_back(msg_part);
                    }
                } else if (!content.is_null()) {
                    throw std::invalid_argument("Invalid 'content' type: expected string or array, got " +
                                                content.dump() +
                                                " (ref: https://github.com/ggml-org/llama.cpp/issues/8367)");
                }
            }
            if (has_tool_calls) {
                for (const auto & tool_call : message.at("tool_calls")) {
                    common_chat_tool_call tc;
                    if (!tool_call.contains("type")) {
                        throw std::invalid_argument("Missing tool call type: " + tool_call.dump());
                    }
                    const auto & type = tool_call.at("type");
                    if (type != "function") {
                        throw std::invalid_argument("Unsupported tool call type: " + tool_call.dump());
                    }
                    if (!tool_call.contains("function")) {
                        throw std::invalid_argument("Missing tool call function: " + tool_call.dump());
                    }
                    const auto & fc = tool_call.at("function");
                    if (!fc.contains("name")) {
                        throw std::invalid_argument("Missing tool call name: " + tool_call.dump());
                    }
                    tc.name           = fc.at("name");
                    const auto & args = fc.at("arguments");
                    if (args.is_string()) {
                        tc.arguments = args;
                    } else {
                        tc.arguments = args.dump();
                    }
                    if (tool_call.contains("id")) {
                        tc.id = tool_call.at("id");
                    }
                    msg.tool_calls.push_back(tc);
                }
            }
            if (!has_content && !has_tool_calls) {
                throw std::invalid_argument(
                    "Expected 'content' or 'tool_calls' (ref: https://github.com/ggml-org/llama.cpp/issues/8367 & "
                    "https://github.com/ggml-org/llama.cpp/issues/12279)");
            }
            if (message.contains("reasoning_content")) {
                msg.reasoning_content = message.at("reasoning_content");
            }
            if (message.contains("name")) {
                msg.tool_name = message.at("name");
            }
            if (message.contains("tool_call_id")) {
                msg.tool_call_id = message.at("tool_call_id");
            }

            msgs.push_back(msg);
        }
    } catch (const std::exception & e) {
        // @ngxson : disable otherwise it's bloating the API response
        // printf("%s\n", std::string("; messages = ") + messages.dump(2));
        throw std::runtime_error("Failed to parse messages: " + std::string(e.what()));
    }

    return msgs;
}

struct messages_inp_normalizer {
    const jinja::caps & caps;

    messages_inp_normalizer(const jinja::caps & c) : caps(c) {}

    // handle supports_string_content / supports_typed_content
    // if string=true and array=false, convert array to string
    // if string=false and array=true, convert string to array
    // if both are true, do nothing
    json normalize(const json & messages) {
        bool only_string = caps.supports_string_content && !caps.supports_typed_content;
        bool only_typed  = !caps.supports_string_content && caps.supports_typed_content;
        if ((!only_string && !only_typed) || !messages.is_array()) {
            return messages;
        }
        json normalized = json::array();
        for (const auto & msg : messages) {
            json copy = msg;
            if (copy.contains("content")) {
                json & it = copy.at("content");
                if (only_typed && it.is_string()) {
                    it = json::array({
                        json{
                            {"type", "text"},
                            {"text", it.get<std::string>()},
                        }
                    });
                } else if (only_string && it.is_array()) {
                    it = concat_content_parts(it);
                }
            }
            normalized.push_back(std::move(copy));
        }
        return normalized;
    }

    // join parts with newline, do not add newline before or after media markers
    static std::string concat_content_parts(const json & parts) {
        std::string text;
        bool last_was_media_marker = false;
        for (const auto & part : parts) {
            std::string type = part.value("type", "");
            bool add_new_line = true;
            if (type == "text") {
                add_new_line = !last_was_media_marker && !text.empty();
                last_was_media_marker = false;
            } else if (type == "media_marker") {
                add_new_line = false;
                last_was_media_marker = true;
            } else {
                LOG_WRN("Ignoring content part type: %s\n", type.c_str());
                continue;
            }

            if (add_new_line) {
                text += '\n';
            }

            text += part.value("text", "");
        }
        return text;
    }
};

static json render_message_to_json(const std::vector<common_chat_msg> & msgs, const jinja::caps & c) {
    if (!c.supports_string_content && !c.supports_typed_content) {
        LOG_WRN("%s: Neither string content nor typed content is supported by the template. This is unexpected and may lead to issues.\n", __func__);
    }

    json messages = json::array();
    for (const auto & msg : msgs) {
        messages.push_back(msg.to_json_oaicompat(/* concat_typed_text= */ false));
    }
    return messages_inp_normalizer(c).normalize(messages);
}

// DEPRECATED: only used in tests
json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text) {
    jinja::caps c;
    c.supports_string_content = true;
    c.supports_typed_content = !concat_typed_text;
    return render_message_to_json(msgs, c);
}

json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools) {
    if (tools.empty()) {
        return json();
    }

    auto result = json::array();
    for (const auto & tool : tools) {
        result.push_back({
            { "type",     "function" },
            { "function", {
                { "name", tool.name },
                { "description", tool.description },
                { "parameters", json::parse(tool.parameters) },
            }},
        });
    }
    return result;
}

json common_chat_tool_parameters(const json & function) {
    if (function.contains("parameters")) {
        const auto & params = function.at("parameters");
        if (!params.is_null() && !(params.is_object() && params.empty())) {
            return params;
        }
    }
    return json{{"type", "object"}, {"properties", json::object()}};
}

std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const json & tools) {
    std::vector<common_chat_tool> result;

    try {
        if (!tools.is_null()) {
            if (!tools.is_array()) {
                throw std::invalid_argument("Expected 'tools' to be an array, got " + tools.dump());
            }
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    throw std::invalid_argument("Missing tool type: " + tool.dump());
                }
                const auto & type = tool.at("type");
                if (!type.is_string() || type != "function") {
                    throw std::invalid_argument("Unsupported tool type: " + tool.dump());
                }
                if (!tool.contains("function")) {
                    throw std::invalid_argument("Missing tool function: " + tool.dump());
                }

                const auto & function = tool.at("function");
                result.push_back({
                    /* .name = */ function.at("name"),
                    /* .description = */ function.value("description", ""),
                    /* .parameters = */ function.value("parameters", json::object()).dump(),
                });
            }
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse tools: " + std::string(e.what()) + "; tools = " + tools.dump(2));
    }

    return result;
}

common_chat_continuation common_chat_continuation_parse(const common_json & value) {
    if (value.is_boolean() && value.get<bool>()) {
        return COMMON_CHAT_CONTINUATION_AUTO;
    }
    if (value.is_string()) {
        auto value_str = value.get<std::string>();
        if (value_str == "reasoning_content") {
            return COMMON_CHAT_CONTINUATION_REASONING;
        }
        if (value_str == "content") {
            return COMMON_CHAT_CONTINUATION_CONTENT;
        }
    }
    return COMMON_CHAT_CONTINUATION_NONE;
}

bool common_chat_verify_template(const std::string & tmpl, bool use_jinja) {
    if (use_jinja) {
        try {
            common_chat_msg msg;
            msg.role    = "user";
            msg.content = "test";

            auto tmpls = common_chat_templates_init(/* model= */ nullptr, tmpl);

            common_chat_templates_inputs inputs;
            inputs.messages = { msg };

            common_chat_templates_apply(tmpls.get(), inputs);
            return true;
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to apply template: %s\n", __func__, e.what());
            return false;
        }
    }
    llama_chat_message chat[] = {
        { "user", "test" }
    };
    const int res = llama_chat_apply_template(tmpl.c_str(), chat, 1, true, nullptr, 0);
    return res >= 0;
}

std::string common_chat_format_single(const struct common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & past_msg,
                                      const common_chat_msg &              new_msg,
                                      bool                                 add_ass,
                                      bool                                 use_jinja) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_bos   = tmpls->add_bos;
    inputs.add_eos   = tmpls->add_eos;

    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages              = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg                 = common_chat_templates_apply(tmpls, inputs).prompt;
    }
    std::ostringstream ss;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;
    auto fmt_new_msg             = common_chat_templates_apply(tmpls, inputs).prompt;
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string common_chat_format_example(const struct common_chat_templates *       tmpls,
                                       bool                                       use_jinja,
                                       const std::map<std::string, std::string> & chat_template_kwargs) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja            = use_jinja;
    inputs.add_bos              = tmpls->add_bos;
    inputs.add_eos              = tmpls->add_eos;
    inputs.chat_template_kwargs = chat_template_kwargs;
    auto add_simple_msg         = [&](auto role, auto content) {
        common_chat_msg msg;
        msg.role    = role;
        msg.content = content;
        inputs.messages.push_back(msg);
    };
    add_simple_msg("system", "You are a helpful assistant");
    add_simple_msg("user", "Hello");
    add_simple_msg("assistant", "Hi there");
    add_simple_msg("user", "How are you?");
    return common_chat_templates_apply(tmpls, inputs).prompt;
}

#define CHATML_TEMPLATE_SRC                                                               \
    "{%- for message in messages -%}\n"                                                   \
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
    "{%- endfor -%}\n"                                                                    \
    "{%- if add_generation_prompt -%}\n"                                                  \
    "  {{- '<|im_start|>assistant\n' -}}\n"                                               \
    "{%- endif -%}"

void common_chat_templates_free(struct common_chat_templates * tmpls) {
    delete tmpls;
}

bool common_chat_templates_was_explicit(const struct common_chat_templates * tmpls) {
    return tmpls->has_explicit_template;
}

common_chat_prompt_preset common_chat_get_asr_prompt(const common_chat_templates * chat_templates) {
    common_chat_prompt_preset asr_preset;
    asr_preset.system = "";
    asr_preset.user   = "Transcribe audio to text";

    if (chat_templates && chat_templates->template_default && is_lfm2_template(chat_templates->template_default->source())) {
        asr_preset.system = "Perform ASR.";
        asr_preset.user   = "";
    }

    return asr_preset;
}

std::string common_chat_templates_source(const struct common_chat_templates * tmpls, const std::string & variant) {
    if (!variant.empty()) {
        if (variant == "tool_use") {
            if (tmpls->template_tool_use) {
                return tmpls->template_tool_use->source();
            }
            return "";
        }
        LOG_DBG("%s: unknown template variant: %s\n", __func__, variant.c_str());
    }
    return tmpls->template_default->source();
}

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override,
                                                     const std::string &        eos_token_override) {
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
    if (chat_template_override.empty()) {
        GGML_ASSERT(model != nullptr);
        const auto * str = llama_model_chat_template(model, /* name */ nullptr);
        if (str) {
            default_template_src  = str;
            has_explicit_template = true;
        }
        str = llama_model_chat_template(model, /* name */ "tool_use");
        if (str) {
            template_tool_use_src = str;
            has_explicit_template = true;
        }
    } else {
        default_template_src = chat_template_override;
    }
    if (default_template_src.empty() || default_template_src == "chatml") {
        if (!template_tool_use_src.empty()) {
            default_template_src = template_tool_use_src;
        } else {
            default_template_src = CHATML_TEMPLATE_SRC;
        }
    }

    // TODO @ngxson : this is a temporary hack to prevent chat template from throwing an error
    // Ref: https://github.com/ggml-org/llama.cpp/pull/15230#issuecomment-3173959633
    if (default_template_src.find("<|channel|>") != std::string::npos
        // search for the error message and patch it
        && default_template_src.find("in message.content or") != std::string::npos) {
        string_replace_all(default_template_src,
                           "{%- if \"<|channel|>analysis<|message|>\" in message.content or "
                           "\"<|channel|>final<|message|>\" in message.content %}",
                           "{%- if false %}");
    }

    // TODO @aldehir : this is a temporary fix, pending Minja changes
    // Ref: https://github.com/ggml-org/llama.cpp/pull/17713#issuecomment-3631342664
    if (default_template_src.find("[TOOL_CALLS]") != std::string::npos
        // search for the error message and patch it
        && default_template_src.find("if (message['content'] is none or") != std::string::npos) {
        string_replace_all(default_template_src,
                           "{%- if (message['content'] is none or message['content'] == '' or "
                           "message['content']|length == 0) and (message['tool_calls'] is not defined or "
                           "message['tool_calls'] is none or message['tool_calls']|length == 0) %}",
                           "{%- if false %}");
    }

    std::string token_bos = bos_token_override;
    std::string token_eos = eos_token_override;
    bool        add_bos   = false;
    bool        add_eos   = false;
    if (model) {
        const auto * vocab     = llama_model_get_vocab(model);
        const auto   get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
            if (token == LLAMA_TOKEN_NULL) {
                if (default_template_src.find(jinja_variable_name) != std::string::npos ||
                    template_tool_use_src.find(jinja_variable_name) != std::string::npos) {
                    LOG_WRN(
                        "common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't "
                          "work as intended.\n",
                        name);
                }
                return std::string();
            }
            return common_token_to_piece(vocab, token, true);
        };
        token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
        token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
        add_bos   = llama_vocab_get_add_bos(vocab);
        add_eos   = llama_vocab_get_add_eos(vocab);
    }
    common_chat_templates_ptr tmpls(new common_chat_templates());
    tmpls->has_explicit_template = has_explicit_template;
    tmpls->add_bos               = add_bos;
    tmpls->add_eos               = add_eos;
    try {
        tmpls->template_default = std::make_unique<common_chat_template>(default_template_src, token_bos, token_eos);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        LOG_ERR("%s: failed to initialize chat template\n", __func__);
        LOG_ERR("%s: please consider disabling jinja via --no-jinja, or using another chat template\n", __func__);
        throw e;
    }
    if (!template_tool_use_src.empty()) {
        try {
            tmpls->template_tool_use = std::make_unique<common_chat_template>(template_tool_use_src, token_bos, token_eos);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to parse tool use chat template (ignoring it): %s\n", __func__, e.what());
        }
    }
    return tmpls;
}

const char * common_chat_format_name(common_chat_format format) {
    switch (format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            return "Content-only";
        case COMMON_CHAT_FORMAT_PEG_SIMPLE:
            return "peg-simple";
        case COMMON_CHAT_FORMAT_PEG_NATIVE:
            return "peg-native";
        case COMMON_CHAT_FORMAT_PEG_GEMMA4:
            return "peg-gemma4";
        case COMMON_CHAT_FORMAT_PEG_MINIMAX_M3:
            return "peg-minimax-m3";
        default:
            throw std::runtime_error("Unknown chat format");
    }
}

const char * common_reasoning_format_name(common_reasoning_format format) {
    switch (format) {
        case COMMON_REASONING_FORMAT_NONE:
            return "none";
        case COMMON_REASONING_FORMAT_AUTO:
            return "auto";
        case COMMON_REASONING_FORMAT_DEEPSEEK:
            return "deepseek";
        case COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY:
            return "deepseek-legacy";
        default:
            throw std::runtime_error("Unknown reasoning format");
    }
}

common_reasoning_format common_reasoning_format_from_name(const std::string & format) {
    if (format == "none") {
        return COMMON_REASONING_FORMAT_NONE;
    }
    if (format == "auto") {
        return COMMON_REASONING_FORMAT_AUTO;
    }
    if (format == "deepseek") {
        return COMMON_REASONING_FORMAT_DEEPSEEK;
    }
    if (format == "deepseek-legacy") {
        return COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
    }
    throw std::runtime_error("Unknown reasoning format: " + format);
}

std::string common_chat_template_direct_apply_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override,
    const std::optional<json> & tools_override,
    const std::optional<json> & additional_context) {
    jinja::context ctx(tmpl.source());

    // messages_override is already built for this template, do not touch its content parts
    json inp = json{
        {"messages", messages_override.has_value()
            ? *messages_override
            : messages_inp_normalizer(tmpl.original_caps()).normalize(inputs.messages)},
        {"bos_token", tmpl.bos_token()},
        {"eos_token", tmpl.eos_token()},
        {"enable_thinking", inputs.enable_thinking},
    };
    if (tools_override.has_value() || !inputs.tools.empty()) {
        inp["tools"] = tools_override.has_value() ? *tools_override : inputs.tools;
    }
    if (inputs.extra_context.is_object()) {
        // TODO: do we need to merge, or replacing is fine?
        for (const auto & [k, v] : inputs.extra_context.items()) {
            inp[k] = v;
        }
    }
    if (additional_context.has_value()) {
        // TODO: merge properly instead of overwriting (matching old behavior)
        for (const auto & [k, v] : additional_context->items()) {
            inp[k] = v;
        }
    }
    if (inputs.add_generation_prompt) {
        inp["add_generation_prompt"] = true;
    }
    if (inp.contains("preserve_reasoning") && inp["preserve_reasoning"].is_boolean()) {
        bool enabled = inp["preserve_reasoning"].get<bool>();
        jinja::caps_apply_preserve_reasoning(ctx, enabled);
    }
    if (inp.contains("reasoning_effort") && inp["reasoning_effort"].is_string() && !inp["reasoning_effort"].empty()) {
        std::string reasoning_effort = inp["reasoning_effort"].get<std::string>();
        jinja::caps_apply_reasoning_effort(ctx, reasoning_effort);
    }

    jinja::global_from_json(ctx, inp, inputs.mark_input);

    // render
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl.prog);
    auto parts = jinja::runtime::gather_string_parts(results);

    std::string result = parts->as_string().str();

    // TODO: improve this later
    if (inputs.add_bos && string_starts_with(result, tmpl.bos_token())) {
        result = result.substr(tmpl.bos_token().size());
    }
    if (inputs.add_eos && string_ends_with(result, tmpl.eos_token())) {
        result = result.substr(0, result.size() - tmpl.eos_token().size());
    }
    return result;
}

std::string common_chat_template_direct_apply(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs) {
    return common_chat_template_direct_apply_impl(tmpl, inputs, std::nullopt, std::nullopt, std::nullopt);
}

std::string common_chat_template_generation_prompt_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override,
    const std::optional<json> & tools_override,
    const std::optional<json> & additional_context) {

    autoparser::generation_params params = inputs;
    params.add_generation_prompt = false;
    params.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    std::string no_gen_prompt    = common_chat_template_direct_apply_impl(tmpl, params, messages_override, tools_override, additional_context);
    params.add_generation_prompt = true;
    std::string gen_prompt       = common_chat_template_direct_apply_impl(tmpl, params, messages_override, tools_override, additional_context);

    size_t prefix_len = 0;
    size_t min_size = std::min(no_gen_prompt.size(), gen_prompt.size());
    while (prefix_len < min_size && no_gen_prompt[prefix_len] == gen_prompt[prefix_len]) {
        prefix_len++;
    }
    return gen_prompt.substr(prefix_len);
}

std::string common_chat_template_generation_prompt(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs) {
    return common_chat_template_generation_prompt_impl(tmpl, inputs, std::nullopt, std::nullopt, std::nullopt);
}

namespace workaround {

static void map_developer_role_to_system(json & messages) {
    for (auto & message : messages) {
        if (message.contains("role")) {
            if (message["role"] == "developer") {
                message["role"] = "system";
            }
        }
    }
}


// if first message is system and template does not support it, merge it with next message
static void system_message_not_supported(json & messages) {
    if (!messages.empty() && messages.front().at("role") == "system") {
        if (messages.size() > 1) {
            LOG_DBG("Merging system prompt into next message\n");
            auto & first_msg = messages.front();
            auto & second_msg = messages[1];
            second_msg["content"] = first_msg.at("content").get<std::string>()
                + "\n" + second_msg.at("content").get<std::string>();
            messages.erase(0);
        } else {
            LOG_WRN("Removing system prompt due to template not supporting system role\n");
            messages.erase(0);
        }
    }
}

static void requires_non_null_content(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls") && !message.contains("content")) {
            message["content"] = "";
        }
    }
}

static void func_args_not_string(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls")) {
            for (auto & tool_call : message["tool_calls"]) {
                if (tool_call.contains("function") && tool_call["function"].contains("arguments")) {
                    auto & args = tool_call["function"]["arguments"];
                    if (args.is_string()) {
                        try {
                            args = json::parse(args.get<std::string>());
                        } catch (const std::exception & e) {
                            throw std::runtime_error("Failed to parse tool call arguments as JSON: " + std::string(e.what()));
                        }
                    }
                }
            }
        }
    }
}

// Trim leading/trailing whitespace from message contents before rendering. This
// has to run on the messages (not on the rendered JSON) because templates with
// string-only content caps concatenate typed content parts into a single string
// during rendering, after which the per-part whitespace can no longer be reached.
// Both the plain string content and the text of typed content parts are trimmed.
static void trim_all_content(std::vector<common_chat_msg> & messages) {
    for (auto & message : messages) {
        message.content           = trim_whitespace(message.content);
        message.reasoning_content = trim_whitespace(message.reasoning_content);
        for (auto & part : message.content_parts) {
            if (part.type == "text") {
                part.text = trim_whitespace(part.text);
            }
        }
    }
}

}

static json common_chat_extra_context() {
    json ctx = json::object();
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::string datetime_str = format_time(now, "%b %d %Y");
    std::string date_str = format_time(now, "%d %b %Y");
    ctx["datetime"] = datetime_str;
    ctx["date_string"] = date_str;
    return ctx;
}

std::optional<common_chat_params> common_chat_try_specialized_template(
        const common_chat_template &          tmpl,
        const std::string &                   src,
        autoparser::generation_params & params) {
    // Ministral/Mistral Large 3 - uses special reasoning structure fixes, can't use autoparser
    // Note: Mistral Small 3.2 uses [CALL_ID] which Ministral doesn't have, so we can distinguish them
    if (src.find("[SYSTEM_PROMPT]") != std::string::npos && src.find("[TOOL_CALLS]") != std::string::npos &&
        src.find("[ARGS]") != std::string::npos && src.find("[CALL_ID]") == std::string::npos) {
        LOG_DBG("Using specialized template: Ministral/Magistral Large 3\n");
        return common_chat_params_init_ministral_3(tmpl, params);
    }

    // GPT-OSS - has unique channel-based structure that needs dedicated handler
    if (src.find("<|channel|>") != std::string::npos) {
        LOG_DBG("Using specialized template: GPT-OSS\n");
        return common_chat_params_init_gpt_oss(tmpl, params);
    }

    // Muse Glimmer format using " to=<recipient>" recipients and <|eom|>/<|eot|> message terminators.
    if (src.find("<atem:function_calls>") != std::string::npos && src.find("<|eom|>") != std::string::npos) {
        LOG_DBG("Using specialized template: Muse Glimmer\n");
        return common_chat_params_init_muse_glimmer(tmpl, params);
    }

    // Functionary v3.2 - uses recipient-based format with >>>recipient\n{content}
    // Detection: template has ">>>all" for content and ">>>" prefix for tool calls
    if (src.find(">>>all") != std::string::npos && src.find(">>>${recipient}") != std::string::npos) {
        LOG_DBG("Using specialized template: Functionary v3.2\n");
        return common_chat_params_init_functionary_v3_2(tmpl, params);
    }

    // Kimi K2 Thinking - uses unique tool call ID format: functions.<name>:<index>
    // Detection: template has "<|tool_calls_section_begin|>" and "functions." prefix in tool call IDs
    if (src.find("<|tool_calls_section_begin|>") != std::string::npos &&
        src.find("<|tool_call_begin|>") != std::string::npos) {
        LOG_DBG("Using specialized template: Kimi K2 Thinking\n");
        return common_chat_params_init_kimi_k2(tmpl, params);
    }

    // Kimi K3 - the <|open|>/<|close|>/<|end_of_msg|> markers are unique to it
    if (src.find("<|open|>") != std::string::npos && src.find("<|close|>") != std::string::npos &&
        src.find("<|end_of_msg|>") != std::string::npos) {
        LOG_DBG("Using specialized template: Kimi K3\n");
        return common_chat_params_init_kimi_k3(tmpl, params);
    }

    // Ling 3.0 / Bailing V3 - <role>X</role> sections with <arg_key>/<arg_value> tagged
    // tool calls. <role> sections are unique to this family among the tagged-arg templates.
    if (src.find("<role>ASSISTANT</role>") != std::string::npos &&
        src.find("<arg_key>") != std::string::npos) {
        LOG_DBG("Using specialized template: Ling 3.0 (Bailing V3)\n");
        return common_chat_params_init_ling3(tmpl, params);
    }

    // Cohere2 MoE / North Code - marker-wrapped format with <|START_TEXT|> content and
    // <|START_ACTION|> JSON tool calls. <|START_TEXT|> is unique to this template (the older
    // Command-R templates use <|START_RESPONSE|>).
    if (src.find("<|START_TEXT|>") != std::string::npos &&
        src.find("<|START_ACTION|>") != std::string::npos) {
        LOG_DBG("Using specialized template: Cohere2 MoE\n");
        return common_chat_params_init_cohere2moe(tmpl, params);
    }

    if (is_lfm2_template(src)) {
        LOG_DBG("Using specialized template: LFM2\n");
        return common_chat_params_init_lfm2(tmpl, params, /* tool_list_tokens = */ true);
    }

    // LFM2.5 format detection: template uses plain "List of tools: [...]" with no special tokens
    if (src.find("List of tools: [") != std::string::npos &&
        src.find("<|tool_list_start|>") == std::string::npos) {
        LOG_DBG("Using specialized template: LFM2.5\n");
        return common_chat_params_init_lfm2(tmpl, params, /* tool_list_tokens = */ false);
    }

    // GigaChatV3 format detection
    if (src.find("<|role_sep|>") != std::string::npos &&
        src.find("<|message_sep|>") != std::string::npos &&
        src.find("<|function_call|>") == std::string::npos) {
        LOG_DBG("Using specialized template: GigaChatV3\n");
        return common_chat_params_init_gigachat_v3(tmpl, params);
    }

    // MiniMax-M3: the namespace token "]<]minimax[>[" collides with the autoparser's
    // markup delimiters, so detect the template and use a dedicated parser.
    if (src.find("]<]minimax[>[") != std::string::npos &&
        src.find("<tool_call>") != std::string::npos &&
        src.find("<invoke name=") != std::string::npos) {
        LOG_DBG("Using specialized template: MiniMax-M3\n");
        return common_chat_params_init_minimax_m3(tmpl, params);
    }

    // DeepSeek V3.2/V4 format detection: template defines dsml_token and uses it for tool calls.
    // The template source contains the token as a variable assignment, not as a literal in markup.
    // V3.2 names the tool call block "function_calls", V4 names it "tool_calls".
    if (src.find("dsml_token") != std::string::npos &&
        src.find("DSML") != std::string::npos &&
        (src.find("function_calls") != std::string::npos ||
         src.find("tool_calls") != std::string::npos)) {
        LOG_DBG("Using specialized template: DeepSeek V3.2/V4\n");
        return common_chat_params_init_deepseek_v3_2(tmpl, params);
    }

    // Gemma4 format detection
    if (src.find("'<|tool_call>call:'") != std::string::npos) {
        if (src.find("{#- OpenAI Chat Completions:") == std::string::npos) {
            // apply workarounds if using the older gemma4 templates
            LOG_WRN("%s: detected an outdated gemma4 chat template, applying compatibility workarounds. "
                    "Consider updating to the official template.\n", __func__);
            workaround::convert_tool_responses_gemma4(params.messages);
        }
        return common_chat_params_init_gemma4(tmpl, params);
    }

    // MiniCPM5 - XML tool calls with <function name="..."><param name="...">...</param></function>
    if (src.find("Tool usage guidelines:") != std::string::npos &&
        src.find("<function name=\"") != std::string::npos &&
        src.find("<param name=\"") != std::string::npos) {
        LOG_DBG("Using specialized template: MiniCPM5\n");
        return common_chat_params_init_minicpm5(tmpl, params);
    }

    // Qwen3-Coder XML tool calls, also used by Nemotron Nano 3, Qwen3.5 and StepFun-3.5-Flash
    if (src.find("<tool_call>") != std::string::npos &&
        src.find("<function=") != std::string::npos &&
        src.find("<parameter=") != std::string::npos &&
        // Exclude models that don't use \n between tags
        src.find("'<tool_call><function=' ~ tool_call.name ~ '>'") == std::string::npos) {
        LOG_DBG("Using specialized template: Qwen3-Coder\n");
        return common_chat_params_init_qwen3_coder(tmpl, params);
    }

    return std::nullopt;
}

static common_chat_params common_chat_templates_apply_jinja(const struct common_chat_templates *        tmpls,
                                                            const struct common_chat_templates_inputs & inputs) {
    autoparser::generation_params params;
    params.tools = common_chat_tools_to_json_oaicompat(inputs.tools);
    const auto & tmpl =
        params.tools.is_array() && tmpls->template_tool_use ? *tmpls->template_tool_use : *tmpls->template_default;
    const auto & src             = tmpl.source();
    const auto & caps            = tmpl.original_caps();
    std::vector<common_chat_msg>        trimmed_messages;
    const std::vector<common_chat_msg> * messages_to_render = &inputs.messages;
    if (src.find("You have access to the following functions in JSONSchema format") != std::string::npos) {
        // StepFun: trim message contents (including typed content parts) before rendering,
        // otherwise leftover whitespace drives the model into reasoning loops (issue #24181)
        trimmed_messages   = inputs.messages;
        workaround::trim_all_content(trimmed_messages);
        messages_to_render = &trimmed_messages;
    }
    params.messages              = render_message_to_json(*messages_to_render, tmpl.original_caps());
    params.tool_choice           = inputs.tool_choice;
    params.reasoning_format      = inputs.reasoning_format;
    params.enable_thinking       = inputs.enable_thinking;
    params.grammar               = inputs.grammar;
    params.now                   = inputs.now;
    params.add_generation_prompt = inputs.add_generation_prompt;
    params.add_bos               = tmpls->add_bos;
    params.add_eos               = tmpls->add_eos;

    params.continue_final_message = inputs.continue_final_message;
    if (params.continue_final_message != COMMON_CHAT_CONTINUATION_NONE) {
        params.add_generation_prompt = false;

        if (!inputs.messages.empty()) {
            // Render messages[:-1] and store continuation message separately
            params.continue_msg = inputs.messages.back();
            params.messages.erase(params.messages.size() - 1);
        }

        if (params.continue_final_message == COMMON_CHAT_CONTINUATION_AUTO && !inputs.messages.empty()) {
            // Resolve based on message content
            params.continue_final_message = COMMON_CHAT_CONTINUATION_CONTENT;
            if (!params.continue_msg.reasoning_content.empty() &&
                params.continue_msg.content.empty() &&
                params.continue_msg.content_parts.empty()) {
                params.continue_final_message = COMMON_CHAT_CONTINUATION_REASONING;
            }
        }
    }

    if (src.find("<|channel|>") == std::string::npos) {
        // map developer to system for all models except for GPT-OSS
        workaround::map_developer_role_to_system(params.messages);
    }

    if (!tmpl.original_caps().supports_system_role) {
        workaround::system_message_not_supported(params.messages);
    }

    if (tmpl.original_caps().supports_tool_calls) {
        // some templates will require the content field in tool call messages
        // to still be non-null, this puts an empty string everywhere where the
        // content field is null
        workaround::requires_non_null_content(params.messages);
    }

    if (tmpl.original_caps().supports_object_arguments) {
        workaround::func_args_not_string(params.messages);
    }

    params.extra_context = common_chat_extra_context();
    for (auto el : inputs.chat_template_kwargs) {
        params.extra_context[el.first] = json::parse(el.second);
    }

    if (!inputs.json_schema.empty()) {
        params.json_schema = json::parse(inputs.json_schema);
    }

    params.parallel_tool_calls = inputs.parallel_tool_calls;

    if (params.tools.is_array()) {
        if (params.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !params.grammar.empty()) {
            throw std::runtime_error("Cannot specify grammar with tools");
        }
        if (caps.supports_tool_calls && !caps.supports_tools) {
            LOG_WRN(
                "Template supports tool calls but does not natively describe tools. The fallback behaviour used may "
                "produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n");
        }
    }

    if (inputs.force_pure_content) {
        LOG_WRN("Forcing pure content template, will not render reasoning or tools separately.");
        // Create the result structure
        common_chat_params data;
        auto params_copy               = params;
        params_copy.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
        data.prompt                    = common_chat_template_direct_apply_impl(tmpl, params_copy);
        data.generation_prompt         = common_chat_template_generation_prompt_impl(tmpl, params);
        data.format                    = COMMON_CHAT_FORMAT_PEG_NATIVE;
        auto parser                    = build_chat_peg_parser([&data](common_chat_peg_builder &p) {
            return p.literal(data.generation_prompt) << p.content(p.rest());
        });
        data.parser                    = parser.save();
        return data;
    }

    if (auto result = common_chat_try_specialized_template(tmpl, src, params)) {
        return *result;
    }

    try {
        LOG_DBG("%s: using differential autoparser\n", __func__);
        struct autoparser::autoparser autoparser;
        autoparser.analyze_template(tmpl);
        auto auto_params = autoparser::peg_generator::generate_parser(tmpl, params, autoparser);

        common_chat_msg_delimiters delimiters;
        if (!autoparser.assistant_start.empty()) {
            delimiters.add(COMMON_CHAT_ROLE_ASSISTANT, autoparser.assistant_start);
        }
        if (!autoparser.user_start.empty()) {
            delimiters.add(COMMON_CHAT_ROLE_USER, autoparser.user_start);
        }

        auto_params.message_delimiters = std::move(delimiters);

        auto_params.supports_thinking = autoparser.reasoning.mode != autoparser::reasoning_mode::NONE;
        if (auto_params.supports_thinking) {
            auto_params.thinking_start_tag = trim_whitespace(autoparser.reasoning.start);
            auto end_tag = trim_whitespace(autoparser.reasoning.end);
            if (!end_tag.empty()) {
                auto_params.thinking_end_tags = {std::move(end_tag)};
            }
        }
        common_peg_arena arena;
        arena.load(auto_params.parser);
        LOG_DBG("%s: generated parser:\n%s\n\nparser generation prompt: %s\n", __func__, arena.dump(arena.root()).c_str(), auto_params.generation_prompt.c_str());
        return auto_params;
    } catch (const std::exception & e) {
        throw std::invalid_argument(std::string("Unable to generate parser for this template. Automatic parser generation failed: ") + e.what());
    }
}

// Legacy template route (adhoc C++ implementation of known templates), forward to llama_chat_apply_template.
static common_chat_params common_chat_templates_apply_legacy(const struct common_chat_templates *        tmpls,
                                                             const struct common_chat_templates_inputs & inputs) {
    size_t                          alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string>        contents;

    for (const auto & msg : inputs.messages) {
        auto content = msg.content;
        for (const auto & part : msg.content_parts) {
            if (part.type != "text" && part.type != "media_marker") {
                LOG_WRN("Ignoring non-text content part: %s\n", part.type.c_str());
                continue;
            }
            if (!content.empty()) {
                content += "\n";
                ;
            }
            content += part.text;
        }
        contents.emplace_back(std::move(content));
    }
    for (size_t i = 0; i < contents.size(); ++i) {
        const auto & msg     = inputs.messages[i];
        const auto & content = contents[i];
        chat.push_back({ msg.role.c_str(), content.c_str() });
        size_t msg_size = msg.role.size() + content.size();
        alloc_size += msg_size + (msg_size / 4);  // == msg_size * 1.25 but avoiding float ops
    }

    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    const auto & src = tmpls->template_default->source();
    int32_t      res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt,
                                                 buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        // if the custom "tmpl" is not supported, we throw an error
        // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
        throw std::runtime_error("this custom template is not supported, try using --jinja");
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt, buf.data(),
                                        buf.size());
    }

    // for safety, we check the result again
    if (res < 0 || (size_t) res > buf.size()) {
        throw std::runtime_error("failed to apply chat template, try using --jinja");
    }

    common_chat_params params;
    params.prompt = std::string(buf.data(), res);
    if (!inputs.json_schema.empty()) {
        params.grammar = json_schema_to_grammar(json::parse(inputs.json_schema));
    } else {
        params.grammar = inputs.grammar;
    }
    return params;
}

common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                               const struct common_chat_templates_inputs & inputs) {
    GGML_ASSERT(tmpls != nullptr);
    return inputs.use_jinja ? common_chat_templates_apply_jinja(tmpls, inputs) :
                              common_chat_templates_apply_legacy(tmpls, inputs);
}

common_chat_msg common_chat_parse(const std::string &               input,
                                  bool                              is_partial,
                                  const common_chat_parser_params & params) {
    return common_chat_peg_parse(params.parser, input, is_partial, params);
}

common_chat_msg common_chat_peg_parse(const common_peg_arena &          src_parser,
                                      const std::string &               input,
                                      bool                              is_partial,
                                      const common_chat_parser_params & params) {
    const common_peg_arena & parser = src_parser.empty() ?
        build_chat_peg_parser([](common_chat_peg_builder & p) { return p.content(p.rest()) + p.end(); }) :
        src_parser;

    if (src_parser.empty()) {
        LOG_DBG("No parser definition detected, assuming pure content parser.");
    }

    const std::string effective_input = params.generation_prompt.empty()
        ? input
        : params.generation_prompt + input;

    //LOG_DBG("Parsing PEG input with format %s: %s\n", common_chat_format_name(params.format), effective_input.c_str());

    common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_LENIENT;
    if (params.debug) {
        flags |= COMMON_PEG_PARSE_FLAG_DEBUG;
    }

    common_peg_parse_context ctx(effective_input, flags);
    auto result = parser.parse(ctx);

    if (result.fail()) {
        // During partial parsing, return partial results if any AST nodes were captured
        // This allows streaming to work correctly for formats like FUNC_MARKDOWN_CODE_BLOCK
        if (is_partial && result.end > 0) {
            // Try to extract any partial results from what was successfully parsed
            common_chat_msg msg;
            msg.role = "assistant";
            std::unique_ptr<common_chat_peg_mapper> mapper;
            if (params.format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
                mapper = std::make_unique<common_chat_peg_gemma4_mapper>(msg);
            } else if (params.format == COMMON_CHAT_FORMAT_PEG_MINIMAX_M3) {
                mapper = std::make_unique<common_chat_peg_minimax_m3_mapper>(msg);
            } else {
                mapper = std::make_unique<common_chat_peg_mapper>(msg);
            }
            mapper->from_ast(ctx.ast, result);

            if (ctx.is_debug()) {
                fprintf(stderr, "\nAST for partial parse (fail):\n%s\n", ctx.ast.dump().c_str());
                fflush(stderr);
            }
            return msg;
        }
        LOG_WRN("%s: unparsed %s output: %s\n", __func__, common_chat_format_name(params.format), effective_input.substr(result.end).c_str());
        LOG_DBG("%s: full %s output triggering error:\n=== BEGIN ===\n%s\n=== END ===\n", __func__, common_chat_format_name(params.format), effective_input.c_str());
        throw std::runtime_error(std::string("The model produced output that does not match the expected ") + common_chat_format_name(params.format) + " format");
    }

    common_chat_msg msg;
    msg.role = "assistant";

    std::unique_ptr<common_chat_peg_mapper> mapper;
    if (params.format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
        mapper = std::make_unique<common_chat_peg_gemma4_mapper>(msg);
    } else if (params.format == COMMON_CHAT_FORMAT_PEG_MINIMAX_M3) {
        mapper = std::make_unique<common_chat_peg_minimax_m3_mapper>(msg);
    } else {
        mapper = std::make_unique<common_chat_peg_mapper>(msg);
    }
    mapper->from_ast(ctx.ast, result);

    if (ctx.is_debug()) {
        fprintf(stderr, "\nAST for %s parse:\n%s\n", is_partial ? "partial" : "full", ctx.ast.dump().c_str());
        fflush(stderr);
    }

    if (!is_partial) {
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat({ msg }).at(0).dump().c_str());
    }
    return msg;
}

std::map<std::string, bool> common_chat_templates_get_caps(const common_chat_templates * chat_templates) {
    GGML_ASSERT(chat_templates != nullptr);
    GGML_ASSERT(chat_templates->template_default != nullptr);
    if (chat_templates->template_tool_use != nullptr) {
        // take the more expressive template when available
        return chat_templates->template_tool_use->caps.to_map();
    }
    return chat_templates->template_default->caps.to_map();
}

// file: common/chat.h
// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

#pragma once

#include "common.h"
#include "peg-parser.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"

#include "json.h"

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <vector>

using chat_template_caps = jinja::caps;

struct common_chat_templates;

namespace autoparser {
struct generation_params;
}  // namespace autoparser

struct common_chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;

    bool operator==(const common_chat_tool_call & other) const {
        return name == other.name && arguments == other.arguments && id == other.id;
    }
};

struct common_chat_msg_content_part {
    std::string type;
    std::string text;

    // TODO @ngxson : no known chat templates support reasoning_content in content parts yet
    //                this can be useful for models with interleaved thinking (like Kimi-K2)
    //                if you see any templates explicitly support this, please ping me
    // std::string reasoning_content;

    bool operator==(const common_chat_msg_content_part & other) const {
        return type == other.type && text == other.text;
    }
};

struct common_chat_template {
    jinja::program prog;
    std::string bos_tok;
    std::string eos_tok;
    std::string src;
    chat_template_caps caps;

    common_chat_template(const std::string & src, const std::string & bos_token, const std::string & eos_token) {
        jinja::lexer lexer;
        auto lexer_res = lexer.tokenize(src);
        this->prog = jinja::parse_from_tokens(lexer_res);

        this->src = lexer_res.source;
        this->bos_tok = bos_token;
        this->eos_tok = eos_token;

        this->caps = jinja::caps_get(prog);
        // LOG_INF("%s: caps:\n%s\n", __func__, this->caps.to_string().c_str());
    }

    const std::string & source() const { return src; }
    const std::string & bos_token() const { return bos_tok; }
    const std::string & eos_token() const { return eos_tok; }

    chat_template_caps original_caps() const {
        return caps;
    }
};

struct common_chat_msg {
    std::string                               role;
    std::string                               content;
    std::vector<common_chat_msg_content_part> content_parts;
    std::vector<common_chat_tool_call>        tool_calls;
    std::string                               reasoning_content;
    std::string                               tool_name;
    std::string                               tool_call_id;

    common_json to_json_oaicompat(bool concat_typed_text = false) const;

    std::string render_content(const std::string & delimiter = "\n\n") const;

    bool empty() const {
        return content.empty() && content_parts.empty() && tool_calls.empty() && reasoning_content.empty() &&
               tool_name.empty() && tool_call_id.empty();
    }

    bool contains_media() const {
        for (const auto & part : content_parts) {
            if (part.type == "media_marker") {
                return true;
            }
        }
        return false;
    }

    void set_tool_call_ids(std::vector<std::string> &           ids_cache,
                           const std::function<std::string()> & gen_tool_call_id) {
        for (auto i = 0u; i < tool_calls.size(); i++) {
            if (ids_cache.size() <= i) {
                auto id = tool_calls[i].id;
                if (id.empty()) {
                    id = gen_tool_call_id();
                }
                ids_cache.push_back(id);
            }
            tool_calls[i].id = ids_cache[i];
        }
    }

    bool operator==(const common_chat_msg & other) const {
        return role == other.role && content == other.content && content_parts == other.content_parts &&
               tool_calls == other.tool_calls && reasoning_content == other.reasoning_content &&
               tool_name == other.tool_name && tool_call_id == other.tool_call_id;
    }

    bool operator!=(const common_chat_msg & other) const { return !(*this == other); }
};

struct common_chat_msg_diff {
    std::string           reasoning_content_delta;
    std::string           content_delta;
    size_t                tool_call_index = std::string::npos;
    common_chat_tool_call tool_call_delta;

    static std::vector<common_chat_msg_diff> compute_diffs(const common_chat_msg & msg_prv,
                                                           const common_chat_msg & msg_new);

    bool operator==(const common_chat_msg_diff & other) const {
        return content_delta == other.content_delta && tool_call_index == other.tool_call_index &&
               tool_call_delta == other.tool_call_delta;
    }
};

enum common_chat_role {
    COMMON_CHAT_ROLE_UNKNOWN,
    COMMON_CHAT_ROLE_SYSTEM,
    COMMON_CHAT_ROLE_ASSISTANT,
    COMMON_CHAT_ROLE_USER,
    COMMON_CHAT_ROLE_TOOL
};

common_chat_role common_chat_role_from_string(const std::string & role);
const char *     common_chat_role_to_string(common_chat_role role);

struct common_chat_msg_span {
    common_chat_role role = COMMON_CHAT_ROLE_UNKNOWN;
    std::size_t pos = 0;
    std::size_t len = 0;

    bool valid() const {
        return role != COMMON_CHAT_ROLE_UNKNOWN;
    }
};

struct common_chat_msg_spans {
    std::vector<common_chat_msg_span> spans;

    void add(common_chat_role role, size_t pos, size_t len) {
        spans.push_back({ role, pos, len });
    }

    bool is_user_start(int32_t pos) const {
        for (auto it = spans.begin(); it != spans.end(); ++it) {
            if (it->role == COMMON_CHAT_ROLE_USER && pos == (int32_t) it->pos) {
                return true;
            }
        }
        return false;
    }

    int32_t last_user_message_pos() const {
        for (auto it = spans.rbegin(); it != spans.rend(); ++it) {
            if (it->role == COMMON_CHAT_ROLE_USER) {
                return (int32_t) it->pos;
            }
        }
        return -1;
    }
};

struct common_chat_msg_delimiter {
    common_chat_role role = COMMON_CHAT_ROLE_UNKNOWN;
    std::string      delimiter;
    llama_tokens     tokens = {};
};

struct common_chat_msg_delimiters {
    std::vector<common_chat_msg_delimiter> delimiters;

    common_chat_msg_delimiters() = default;
    common_chat_msg_delimiters(std::initializer_list<common_chat_msg_delimiter> delims) : delimiters(delims) {}

    void add(common_chat_role role, const std::string & delimiter) {
        delimiters.push_back({ role, delimiter });
    }

    void tokenize(const llama_vocab * vocab);

    // split tokens into message spans. skips maps a start index to a length of a region to jump over without matching
    common_chat_msg_spans split(const llama_tokens & tokens, const std::map<size_t, size_t> & skips = {}) const;

    common_json to_json() const;
};

struct common_chat_tool {
    std::string name;
    std::string description;
    std::string parameters;
};

enum common_chat_tool_choice {
    COMMON_CHAT_TOOL_CHOICE_AUTO,
    COMMON_CHAT_TOOL_CHOICE_REQUIRED,
    COMMON_CHAT_TOOL_CHOICE_NONE,
};

enum common_chat_format {
    COMMON_CHAT_FORMAT_CONTENT_ONLY,

    // These are intended to be parsed by the PEG parser
    COMMON_CHAT_FORMAT_PEG_SIMPLE,
    COMMON_CHAT_FORMAT_PEG_NATIVE,
    COMMON_CHAT_FORMAT_PEG_GEMMA4,
    COMMON_CHAT_FORMAT_PEG_MINIMAX_M3,

    COMMON_CHAT_FORMAT_COUNT,  // Not a format, just the # formats
};


// Continuation method provided via `continue_final_message`
enum common_chat_continuation {
    COMMON_CHAT_CONTINUATION_NONE,
    COMMON_CHAT_CONTINUATION_AUTO,
    COMMON_CHAT_CONTINUATION_REASONING,
    COMMON_CHAT_CONTINUATION_CONTENT,
};

struct common_chat_templates_inputs {
    std::vector<common_chat_msg>          messages;
    std::string                           grammar;
    std::string                           json_schema;
    bool                                  add_generation_prompt  = true;
    common_chat_continuation              continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    bool                                  use_jinja              = true;
    // Parameters below only supported when use_jinja is true
    std::vector<common_chat_tool>         tools;
    common_chat_tool_choice               tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool                                  parallel_tool_calls = false;
    common_reasoning_format               reasoning_format    = COMMON_REASONING_FORMAT_NONE; // TODO: refactor this to "bool enable_thinking"
    bool                                  enable_thinking     = true;
    std::chrono::system_clock::time_point now                 = std::chrono::system_clock::now();
    std::map<std::string, std::string>    chat_template_kwargs;
    bool                                  add_bos = false;
    bool                                  add_eos = false;
    bool                                  force_pure_content = false;
};

struct common_chat_params {
    common_chat_format                  format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string                         prompt;
    std::string                         grammar;
    bool                                grammar_lazy         = false;
    std::string                         generation_prompt;
    bool                                supports_thinking    = false;
    std::string                         thinking_start_tag;  // e.g., "<think>"
    std::vector<std::string>            thinking_end_tags;   // e.g., "</think>"
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::vector<std::string>            additional_stops;
    std::string                         parser;
    common_chat_msg_delimiters          message_delimiters;
};

// per-message parsing syntax
// should be derived from common_chat_params
struct common_chat_parser_params {
    common_chat_format      format               = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format reasoning_format     = COMMON_REASONING_FORMAT_NONE; // TODO: refactor this to "bool parse_reasoning"
    // Whether reasoning_content should be inlined in the content (e.g. for reasoning_format=deepseek in stream mode)
    bool                    reasoning_in_content = false;
    std::string             generation_prompt;
    bool                    parse_tool_calls     = true;
    bool                    is_continuation      = false;
    bool                    echo                 = false;  // Include assistant prefilled msg in output
    bool                    debug                = false;  // Enable debug output for PEG parser
    common_peg_arena        parser               = {};
    common_chat_parser_params() = default;
    common_chat_parser_params(const common_chat_params & chat_params) {
        format  = chat_params.format;
        generation_prompt = chat_params.generation_prompt;
    }
};

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
bool common_chat_verify_template(const std::string & tmpl, bool use_jinja);

void common_chat_templates_free(struct common_chat_templates * tmpls);

struct common_chat_templates_deleter {
    void operator()(common_chat_templates * tmpls) { common_chat_templates_free(tmpls); }
};

typedef std::unique_ptr<struct common_chat_templates, common_chat_templates_deleter> common_chat_templates_ptr;

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override = "",
                                                     const std::string &        eos_token_override = "");

bool        common_chat_templates_was_explicit(const struct common_chat_templates * tmpls);
std::string common_chat_templates_source(const struct common_chat_templates * tmpls, const std::string & variant = "");

struct common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                                      const struct common_chat_templates_inputs & inputs);

// Format single message, while taking into account the position of that message in chat history
std::string common_chat_format_single(const struct common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & past_msg,
                                      const common_chat_msg &              new_msg,
                                      bool                                 add_ass,
                                      bool                                 use_jinja);

// Returns an example of formatted chat
std::string common_chat_format_example(const struct common_chat_templates *       tmpls,
                                       bool                                       use_jinja,
                                       const std::map<std::string, std::string> & chat_template_kwargs);

const char *    common_chat_format_name(common_chat_format format);
common_chat_msg common_chat_parse(const std::string & input, bool is_partial, const common_chat_parser_params & params);
common_chat_msg common_chat_peg_parse(const common_peg_arena & src_parser, const std::string & input, bool is_partial, const common_chat_parser_params & params);

// used by arg and server
const char *            common_reasoning_format_name(common_reasoning_format format);
common_reasoning_format common_reasoning_format_from_name(const std::string & format);

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice);

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates);

// Parses a JSON array of messages in OpenAI's chat completion API format.
std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const common_json & messages);

std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const common_json & tools);

common_chat_continuation common_chat_continuation_parse(const common_json & value);

// DEPRECATED: only used in tests
common_json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text = false);

common_json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools);

// The parameters schema of a function tool. A tool without parameters, or with an empty {}, takes zero arguments.
common_json common_chat_tool_parameters(const common_json & function);

// get template caps, useful for reporting to server /props endpoint
std::map<std::string, bool> common_chat_templates_get_caps(const common_chat_templates * chat_templates);

std::string common_chat_template_direct_apply(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs);

std::string common_chat_template_generation_prompt(
    const common_chat_template &          tmpl,
    const autoparser::generation_params & inputs);

std::optional<common_chat_params> common_chat_try_specialized_template(
        const common_chat_template &          tmpl,
        const std::string &                   src,
        autoparser::generation_params & params);


// specialized per-task preset
struct common_chat_prompt_preset {
    std::string system;
    std::string user;
};

common_chat_prompt_preset common_chat_get_asr_prompt(const common_chat_templates * chat_templates);

common_chat_msg_delimiters common_chat_msg_delimiters_parse(const common_json & delimiters);

// file: common/common.cpp
#include "ggml.h"
#include "gguf.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/types.h>
#include <pwd.h>
#endif

#if defined(_AIX)
#include <sys/systemcfg.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

common_time_meas::common_time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

common_time_meas::~common_time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

//
// CPU utils
//

int32_t common_cpu_get_num_physical_cores() {
#if defined(_AIX)
    int32_t logical_cpus = _system_configuration.ncpus;
    int32_t smt_threads = _system_configuration.smt_threads;
    if (smt_threads > 0) {
        return static_cast<int32_t>(logical_cpus / smt_threads);
    }
    if (logical_cpus > 0) {
        return static_cast<int32_t>(logical_cpus);
    }
#elif defined(__linux__)
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    unsigned int n_threads_win = std::thread::hardware_concurrency();
    unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return default_threads;
        }
    }

    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
        return default_threads;
    }

    int32_t num_physical_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            num_physical_cores += info->Processor.GroupCount;
        }
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char*>(info) + info->Size);
    }

