using System;
using UnityEngine;
using System.Collections;
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
public class Block
{
    public bool empty;
    public byte id = 0;
    public int health;

    public Block ReturnBlock => this;

    public Block(bool isEmpty)
    {
        empty = isEmpty;
        health = 100;
    }

    public void DamageBlock(int points)
    {
        Debug.Log("Damage to block from " + health + " to " + (health - points));
        health -= points;
        if(health <= 0){
            empty = true;
            Debug.Log("setting to empty");
        }else if (health <= 50) {
            Debug.Log("damaged");
            id += 4;
        }
    }

    public override string ToString() {
        return "Id: " + id + ", empty: " + empty + ", health: " + health;
    }
}