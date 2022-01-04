using System;
using UnityEngine;
using System.Collections;
using Unity.Netcode;
using Object = System.Object;


public enum BlockFace {
    All,
    Top, //Y+
    Bottom, //Y-
    Left, //X-
    Right, //X+
    Far, //Z+
    Near //Z-    
}

public enum BlockType {
    Grass = 0, // Type and corresponding health 
    Earth = 1,
    Metal = 2,
    Stone = 3,
}

// public enum BlockType {
//     Grass = 120, // Type and corresponding health 
//     Stone = 200,
//     Metal = 300,
//     Earth = 100,
// }

public static class BlockUtils {
    public static BlockType IdToBlockType(byte id) {
        switch (id) {
            case 0: case 4:
                return BlockType.Grass;
            case 1: case 5:
                return BlockType.Earth;
            case 2: case 6:
                return BlockType.Metal;
            case 3: case 7:
                return BlockType.Stone;
            default:
                Debug.Log("Error in Block.cs: IdToBlockType: invalid id");
                return BlockType.Grass;
        }
    }
    
    public static byte BlockTypeToId(BlockType bt) {
        switch (bt) {
            case BlockType.Grass:
                return 0;
            case BlockType.Earth:
                return 1;
            case BlockType.Metal:
                return 2;
            case BlockType.Stone:
                return 3;
            default:
                Debug.Log("Error in Block.cs: BlockTypeToId: invalid BlockType");
                return 0;
        }
    }
}


[Serializable]
public class Block : INetworkSerializable {
    public bool Empty {
        get => health == 0;
        set => health = (sbyte) (value ? 0 : 100);
    }

    public byte id;
    public sbyte health;

    public Block() {
    }

    public Block(bool isEmpty) {
        health = 0;
        id = 0;
        Empty = isEmpty;
    }

    public void DamageBlock(sbyte points) {
        Debug.Log("Damage to block from " + health + " to " + (health - points));
        health -= points;
        if (health <= 0) {
            Empty = true;
            Debug.Log("setting to empty");
        }
        else if (health <= 50 && id < 4) {
            Debug.Log("damaged");
            id += 4;
        }
    }

    public override string ToString() {
        return "Id: " + id + ", empty: " + Empty + ", health: " + health;
    }

    public void NetworkSerialize<T>(BufferSerializer<T> serializer) where T : IReaderWriter {
        serializer.SerializeValue(ref id);
        serializer.SerializeValue(ref health);
    }
}