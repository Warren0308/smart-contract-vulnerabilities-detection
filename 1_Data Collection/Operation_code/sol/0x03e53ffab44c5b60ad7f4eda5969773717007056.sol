PUSH1 0x60<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH1 0x3e<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0xe8f35f2c<br>DUP2<br>EQ<br>PUSH1 0x43<br>JUMPI<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x49<br>PUSH1 0x4b<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>CALLER<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>SWAP1<br>DUP2<br>AND<br>SWAP2<br>AND<br>EQ<br>PUSH1 0x72<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>SWAP1<br>DUP2<br>AND<br>SWAP1<br>CALLVALUE<br>ADDRESS<br>SWAP1<br>SWAP2<br>AND<br>BALANCE<br>SUB<br>DUP1<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH1 0xbb<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMP<br>STOP<br>