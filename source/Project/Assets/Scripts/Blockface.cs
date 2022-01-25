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
    Earth = 0,
    Wood = 1,
    Stone = 2,
    Iron = 3,
}


// public enum BlockType {
//     Grass = 120, // Type and corresponding health 
//     Stone = 200,
//     Metal = 300,
//     Earth = 100,
// }

public static class BlockProperties {
    public static sbyte MaxHealth(BlockType bt) {
        switch (bt) {
            case BlockType.Wood:
                return 50;
            case BlockType.Earth:
                return 30;
            case BlockType.Stone:
                return 70;
            case BlockType.Iron:
                return 100;
            default:
                Debug.Log("Error in Blockface.cs: BlockProperties.MaxHealth invalid BlockType");
                return 0;
        }
    }
}


[Serializable]
public class Block : INetworkSerializable {
    public bool Empty {
        get => health == 0;
        set => health = (sbyte) (value ? 0 : MaxHealth);
    }

    public sbyte MaxHealth => BlockProperties.MaxHealth((BlockType) id);

    public byte id;
    public sbyte health;

    public Block() {
    }

    public Block(bool isEmpty, byte id) {
        health = 0;
        this.id = id;
        Empty = isEmpty;
    }

    public void DamageBlock(sbyte points) {
        Debug.Log("Damage to block from " + health + " to " + (health - points));
        health -= points;
        if (health <= 0) {
            Empty = true;
            Debug.Log("setting to empty");
        }
        else if (health <= MaxHealth / 2 && id < 4) {
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