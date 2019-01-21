#pragma once

#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

class Reader {    
 public:
    int nonterm_count = 0;
    int vertices_count = 0;
    std::map<std::string, int> nonterm_to_int;
    std::unordered_map<std::string, std::vector<int>> term_to_nonterm;
    std::vector<std::pair<int, std::pair<int, int>>> nonterm_prods;
    std::vector<std::pair<std::string, std::pair<int, int>>> edges;

    Reader(char* chomsky_file, char* graph_file) {
        auto chomsky_stream = std::ifstream(chomsky_file, std::ifstream::in);

        std::string line, tmp;
        
        while(std::getline(chomsky_stream, line)){
            std::vector<std::string> terms;
            std::istringstream iss(line);
            while(iss >> tmp) {
                terms.push_back(tmp);
            }
            if (!nonterm_to_int.count(terms[0])) {
                nonterm_to_int[terms[0]] = nonterm_count++;
            }
            if (terms.size() == 2) {
                if (!term_to_nonterm.count(terms[1])) {
                    term_to_nonterm[terms[1]] = {};
                }
                term_to_nonterm[terms[1]].push_back(nonterm_to_int[terms[0]]);
            }
            else if (terms.size() == 3) {
                if (!nonterm_to_int.count(terms[1])) {
                    nonterm_to_int[terms[1]] = nonterm_count++;
                }
                if (!nonterm_to_int.count(terms[2])) {
                    nonterm_to_int[terms[2]] = nonterm_count++;
                }
                nonterm_prods.push_back({nonterm_to_int[terms[0]], {nonterm_to_int[terms[1]], nonterm_to_int[terms[2]]}});
            }
        }
        chomsky_stream.close();

        auto graph_stream = std::ifstream(graph_file, std::ifstream::in);
        int from, to;
        std::string terminal;
        while (graph_stream >> from >> terminal >> to) {
            // --from, --to;
            edges.push_back({terminal, {from, to}});
            vertices_count = std::max(vertices_count, std::max(from, to) + 1);
        }
        graph_stream.close();
    }
};