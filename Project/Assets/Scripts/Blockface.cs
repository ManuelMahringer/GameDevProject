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
    public bool empty = false;
    public byte id = 0;

    public Block ReturnBlock { get { return this; } }

    public Block(bool isEmpty)
    {
        empty = isEmpty;
    }
}