using System;
using UnityEngine;
using System.Collections;
using Unity.Netcode;
using Object = System.Object;


public enum BlockFace
{
    All,
    Top, //Y+
    Bottom, //Y-
    Left, //X-
    Right, //X+
    Far, //Z+
    Near //Z-    
}

public enum BlockType
{
    Grass = 100, // Type and corresponding health 
    Stone = 200,
    Metal = 300,
    Earth = 100,
}

[Serializable]
public class Block : INetworkSerializable {
    public bool Empty {
        get => health == 0;
        set => health = (byte) (value ? 0 : 100);
    }
    public byte id;
    public byte health;

    public Block() {}
    
    public Block(bool isEmpty)
    {
        health = 0;
        id = 0;
        Empty = isEmpty;
    }

    public void DamageBlock(byte points)
    {
        Debug.Log("Damage to block from " + health + " to " + (health - points));
        health -= points;
        if(health <= 0){
            Empty = true;
            Debug.Log("setting to empty");
        }else if (health <= 50) {
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