PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x00<br>CALLVALUE<br>ISZERO<br>ISZERO<br>PUSH1 0x96<br>JUMPI<br>POP<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x02<br>ADDRESS<br>BALANCE<br>DIV<br>SWAP1<br>PUSH20 0x6b6e4b338b4d5f7d847dab5492106751c57b7ff0<br>SWAP1<br>PUSH2 0x08fc<br>DUP4<br>ISZERO<br>MUL<br>SWAP1<br>DUP4<br>SWAP1<br>PUSH1 0x00<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH1 0x53<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>PUSH1 0x40<br>MLOAD<br>PUSH20 0xe09f3630663b6b86e82d750b00206f8f8c6f8ad4<br>SWAP1<br>DUP3<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP1<br>DUP4<br>SWAP1<br>PUSH1 0x00<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH1 0x94<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>JUMPDEST<br>POP<br>STOP<br>STOP<br>