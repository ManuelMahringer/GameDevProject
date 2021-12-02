using UnityEngine;
using System.Collections;


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

public class Block
{
    public bool empty;
    public byte id = 0;
    public int health;

    public Block ReturnBlock { get { return this; } }

    public Block(bool isEmpty)
    {
        empty = isEmpty;
        health = 100;
    }

    public void DamageBlock(int points)
    {
        Debug.Log("Damage to block from " + health + " to " + (health - points));
        health -= points;
        if(health <= 0)
        {
            empty = true;
            Debug.Log("setting to empty");
        }
    }
}