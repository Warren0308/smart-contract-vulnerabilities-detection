PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH1 0x47<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH3 0xa36947<br>DUP2<br>EQ<br>PUSH1 0x49<br>JUMPI<br>DUP1<br>PUSH4 0x1b9265b8<br>EQ<br>PUSH1 0x5b<br>JUMPI<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH1 0x54<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH1 0x47<br>PUSH1 0x61<br>JUMP<br>JUMPDEST<br>PUSH1 0x47<br>PUSH1 0x85<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>ORIGIN<br>EQ<br>ISZERO<br>PUSH1 0x83<br>JUMPI<br>ORIGIN<br>SELFDESTRUCT<br>JUMPDEST<br>JUMP<br>JUMPDEST<br>ADDRESS<br>BALANCE<br>CALLVALUE<br>LT<br>PUSH1 0x83<br>JUMPI<br>PUSH1 0x40<br>MLOAD<br>ORIGIN<br>SWAP1<br>ADDRESS<br>BALANCE<br>DUP1<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP2<br>PUSH1 0x00<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH1 0xb8<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>JUMP<br>STOP<br>