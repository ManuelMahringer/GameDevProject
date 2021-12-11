using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading.Tasks;
using Unity.Netcode;
using Random = UnityEngine.Random;


public class Chunk : NetworkBehaviour
{
    public PhysicMaterial worldMaterial;
    Mesh chunkMesh;
    float blockSize = 1;

    RVector3 chunkPosition;
    public RVector3 Position { get { return chunkPosition; } set { chunkPosition = value; } }

    public RVector3 chunkSize;
    public RVector3 Size { get { return chunkSize; } set { chunkSize = value; } }

    public Block[,,] chunkBlocks;
    public Block[,,] ReturnChunkBlocks { get { return chunkBlocks; } }

    public Chunk ThisChunk { get { return this; } }

    public int seed;
    public int intensity;

    List<Vector3> chunkVerticies = new List<Vector3>();
    List<Vector2> chunkUV = new List<Vector2>();
    List<int> chunkTriangles = new List<int>();
    int VerticiesIndex;


    public Vector2 textureBlockSize;
    Texture textureAtlas;
    Vector2 atlasSize;

    public Material highlightBlockMaterial;

    [ServerRpc (RequireOwnership = false)]
    public void PingServerRpc(string sometext) {
        Debug.Log("ServerRPC Called " + sometext); 
        
    }
    
    void Awake()
    {
        textureAtlas = transform.GetComponent<MeshRenderer>().material.mainTexture;

        atlasSize = new Vector2(textureAtlas.width / textureBlockSize.x, textureAtlas.height / textureBlockSize.y);
        chunkMesh = this.GetComponent<MeshFilter>().mesh;
        chunkMesh.MarkDynamic();
        if (IsOwnedByServer) {
            GenerateChunk();
        }
    }


    public void DestroyBlock(Vector3 hit) 
    {
        Debug.Log(hit.x + " " + hit.y + " " + hit.z);
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].empty = true;
        //RemoveVerticies(Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z));      
        UpdateChunk();
    }
    
    [ServerRpc (RequireOwnership = false)]
    public void DestroyBlockServerRpc(Vector3 hit) 
    {
        Debug.Log("DESTROY BLOCK PER RPC");
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].empty = true;
        //RemoveVerticies(Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z));      
        UpdateChunk();
        DestroyBlockClientRpc(hit);
    }
    
    [ClientRpc]
    public void DestroyBlockClientRpc(Vector3 hit) 
    {
        Debug.Log("DESTROY BLOCK PER CLIENT RPC");
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].empty = true;
        //RemoveVerticies(Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z));      
        UpdateChunk();
    }

    public void BuildBlock(Vector3 hit) 
    {
        Debug.Log("NORMAL BUILD BLOCK: " + hit.x + " " + hit.y + " " + hit.z);
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].empty = false;
        UpdateChunk();
        BuildBlockClientRpc(hit);
    }
    [ClientRpc]
    public void BuildBlockClientRpc(Vector3 hit) 
    {
        Debug.Log("Client RPC BUILD BLOCK: " + hit.x + " " + hit.y + " " + hit.z);
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].empty = false;
        UpdateChunk();
    }
    
    [ServerRpc (RequireOwnership = false)]
    public void BuildBlockServerRpc(Vector3 hit) 
    {
        Debug.Log(hit.x + " " + hit.y + " " + hit.z);
        
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].empty = false;
        Debug.Log("BUILDING");
        UpdateChunk();
    }

    public void DamageBlock(Vector3 hit, int damage)
    {
        Debug.Log(hit.x + " " + hit.y + " " + hit.z);
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].DamageBlock(damage);
        Debug.Log("Damaging");
        UpdateChunk();
    }

    public void GenerateChunk()
    {
        float[,] chunkHeights = Noise.Generate(chunkSize.x + 1, chunkSize.y + 1, seed, intensity);
        chunkBlocks = new Block[chunkSize.x + 1, chunkSize.y + 1, chunkSize.z + 1];

        for (int x = 0; x <= chunkSize.x; x++)
        {
            for (int z = 0; z <= chunkSize.z; z++)
            {
                for (int y = 0; y <= chunkSize.y; y++)
                {
                    chunkBlocks[x, y, z] = new Block(true);
                    
                    if (y <= chunkHeights[x, z])
                    {
                        chunkBlocks[x, y, z] = new Block(false);
                        chunkBlocks[x, y, z].id = (byte)Random.Range(0, 4);
                        //Debug.Log("Creating Block with id" + chunkBlocks[x, y, z].id);
                    }
                    
                }
            }
        }
        UpdateChunk();
    }
    private void AddVerticies(int x, int y, int z)
    {
        if (CheckSides(new RVector3(x, y, z), BlockFace.Top))
        {
            VerticiesIndex = chunkVerticies.Count;


            chunkVerticies.Add(new Vector3(x, y + blockSize, z));
            chunkVerticies.Add(new Vector3(x, y + blockSize, z + blockSize));
            chunkVerticies.Add(new Vector3(x + blockSize, y + blockSize, z + blockSize));
            chunkVerticies.Add(new Vector3(x + blockSize, y + blockSize, z));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }

        if (CheckSides(new RVector3(x, y, z), BlockFace.Bottom))
        {
            VerticiesIndex = chunkVerticies.Count;

            chunkVerticies.Add(new Vector3(x, y, z));
            chunkVerticies.Add(new Vector3(x + blockSize, y, z));
            chunkVerticies.Add(new Vector3(x + blockSize, y, z + blockSize));
            chunkVerticies.Add(new Vector3(x, y, z + blockSize));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }
        if (CheckSides(new RVector3(x, y, z), BlockFace.Right))
        {
            VerticiesIndex = chunkVerticies.Count;

            chunkVerticies.Add(new Vector3(x + blockSize, y, z));
            chunkVerticies.Add(new Vector3(x + blockSize, y + blockSize, z));
            chunkVerticies.Add(new Vector3(x + blockSize, y + blockSize, z + blockSize));
            chunkVerticies.Add(new Vector3(x + blockSize, y, z + blockSize));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }

        if (CheckSides(new RVector3(x, y, z), BlockFace.Left))
        {
            VerticiesIndex = chunkVerticies.Count;

            chunkVerticies.Add(new Vector3(x, y, z + blockSize));
            chunkVerticies.Add(new Vector3(x, y + blockSize, z + blockSize));
            chunkVerticies.Add(new Vector3(x, y + blockSize, z));
            chunkVerticies.Add(new Vector3(x, y, z));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }

        if (CheckSides(new RVector3(x, y, z), BlockFace.Far))
        {
            VerticiesIndex = chunkVerticies.Count;

            chunkVerticies.Add(new Vector3(x, y, z + blockSize));
            chunkVerticies.Add(new Vector3(x + blockSize, y, z + blockSize));
            chunkVerticies.Add(new Vector3(x + blockSize, y + blockSize, z + blockSize));
            chunkVerticies.Add(new Vector3(x, y + blockSize, z + blockSize));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }

        if (CheckSides(new RVector3(x, y, z), BlockFace.Near))
        {
            VerticiesIndex = chunkVerticies.Count;

            chunkVerticies.Add(new Vector3(x, y, z));
            chunkVerticies.Add(new Vector3(x, y + blockSize, z));
            chunkVerticies.Add(new Vector3(x + blockSize, y + blockSize, z));
            chunkVerticies.Add(new Vector3(x + blockSize, y, z));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }
    }
    
    public void UpdateChunk()
    {
        //Debug.Log("start updating " + Time.time * 1000);

        chunkVerticies = new List<Vector3>();
        chunkUV = new List<Vector2>();
        chunkTriangles = new List<int>();

        chunkMesh.Clear();

        for (var y = 0; y <= chunkSize.y; y++)
        {
            for (var x = 0; x <= chunkSize.x; x++)
            {
                for (var z = 0; z <= chunkSize.z; z++)
                {
                    if (!chunkBlocks[x, y, z].empty)
                    {
                        AddVerticies(x, y, z);
                    }
                }
            }

        }
        //Debug.Log("before finzalize "+  +Time.time * 1000);
        FinalizeChunk();
        
        Destroy(gameObject.GetComponent<MeshCollider>());
        MeshCollider mc = gameObject.AddComponent<MeshCollider>();
        mc.material = worldMaterial;
        
    
        //Debug.Log("end updating " + Time.time * 1000);
    }

    public bool CheckSides(RVector3 blockPosition, BlockFace blockFace)
    {
        int x, y, z;
        x = blockPosition.x;
        y = blockPosition.y;
        z = blockPosition.z;


        switch (blockFace)
        {

            case BlockFace.Top:    //Checks top face
                if (y + 1 <= chunkSize.y)
                {
                    //Debug.Log(chunkBlocks[x, y + 1, z].empty);
                    if (!chunkBlocks[x, y + 1, z].empty)
                    { return false; }
                }
                break;


            case BlockFace.Bottom:    //Checks bottom face
                if (y - 1 >= 0)
                    if(!chunkBlocks[x, y - 1, z].empty)
                { return false; }
                break;

            case BlockFace.Right:    //Checks right face


                if (x + 1 <= chunkSize.x)
                {
                    if (!chunkBlocks[x + 1, y, z].empty)
                    { return false; }
                }
                break;


            case BlockFace.Left:    //Checks Left face

                if (x - 1 >= 0)
                {
                    if (!chunkBlocks[x - 1, y, z].empty)
                    { return false; }
                }
                break;


            case BlockFace.Far:    //Checks Far face

                if (z + 1 <= chunkSize.z)
                {
                    if (!chunkBlocks[x, y, z + 1].empty)
                    { return false; }
                }
                break;


            case BlockFace.Near:    //Checks Near face

                if (z - 1 >= 0)
                {
                    if (!chunkBlocks[x, y, z - 1].empty)
                    { return false; }
                }
                break;

        }
        return true;
    }

    void UpdateChunkUV(byte blockID)
    { 
        chunkTriangles.Add(VerticiesIndex);
        chunkTriangles.Add(VerticiesIndex + 1);
        chunkTriangles.Add(VerticiesIndex + 2);

        chunkTriangles.Add(VerticiesIndex + 2);
        chunkTriangles.Add(VerticiesIndex + 3);
        chunkTriangles.Add(VerticiesIndex);

        Vector2 textureInterval = new Vector2(1 / atlasSize.x, 1 / atlasSize.y);

        Vector2 textureID = new Vector2(textureInterval.x * (blockID % atlasSize.x), textureInterval.y * Mathf.FloorToInt(blockID / (2 * atlasSize.y)));
        
        
        /*.Log("texID " + textureID);
        Debug.Log("atlasid x " + atlasSize.x + " y " + atlasSize.y);
        Debug.Log("HÄÄ" +textureInterval.x * (blockID % atlasSize.x));
        Debug.Log("textureInterval" + textureInterval);*/
        chunkUV.Add(new Vector2(textureID.x, textureID.y - textureInterval.y));
        chunkUV.Add(new Vector2(textureID.x + textureInterval.x, textureID.y - textureInterval.y));
        chunkUV.Add(new Vector2(textureID.x + textureInterval.x, textureID.y));
        chunkUV.Add(new Vector2(textureID.x, textureID.y));
    }

    void FinalizeChunk()
    {
        chunkMesh.vertices = chunkVerticies.ToArray();
        chunkMesh.triangles = chunkTriangles.ToArray();
        chunkMesh.uv = chunkUV.ToArray();
        chunkMesh.RecalculateNormals();
    }
}
