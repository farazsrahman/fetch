#include "tamols/tamols_state.h"
#include <cassert>
#include <iostream>

int main() {
    try {
        // Create state
        tamols::TAMOLSState state;
        
        // Basic checks
        assert(state.num_legs == 4);
        assert(state.base_dims == 6);
        
        // Test program creation - ERRS
        state.setupVariables();
        
        // // Since prog is an object, we should check if it is initialized correctly
        // // by checking its properties or methods, not comparing to nullptr.
        // assert(state.prog.num_vars() > 0);
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        std::cerr << "Error type: " << typeid(e).name() << std::endl;
        std::cerr << "Error message: " << e.what() << std::endl;
        return 1;
    }
}