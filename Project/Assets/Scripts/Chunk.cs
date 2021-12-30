using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;
using System.Threading.Tasks;
using Unity.Netcode;
using UnityEditor;
using Random = UnityEngine.Random;


public class Chunk : NetworkBehaviour {
    public PhysicMaterial worldMaterial;
    public float blockSize = 1;
    public RVector3 chunkSize;
    public int seed;
    public int intensity;
    public Vector2 textureBlockSize;

    private Block[,,] chunkBlocks;
    private Mesh chunkMesh;
    private List<Vector3> chunkVertices = new List<Vector3>();
    private List<Vector2> chunkUV = new List<Vector2>();
    private List<int> chunkTriangles = new List<int>();
    private int verticesIndex;
    private Texture textureAtlas;
    private Vector2 atlasSize;

    void Awake() {
        textureAtlas = transform.GetComponent<MeshRenderer>().material.mainTexture;

        atlasSize = new Vector2(textureAtlas.width / textureBlockSize.x, textureAtlas.height / textureBlockSize.y);
        chunkMesh = GetComponent<MeshFilter>().mesh;
        chunkMesh.MarkDynamic();
        if (IsOwnedByServer) {
            if (ComponentManager.Map.Name == "Generate") {
                // dummy option to still be able to generate the random map TODO: remove
                GenerateChunk();
            }
        }
    }

    [ClientRpc]
    public void ReceiveInitialChunkDataClientRpc(Block[] blocks) {
        Debug.Log("CHUNK BLOCKS RECEIVED");
        chunkBlocks = ExpandBlocks(chunkSize.x+1, chunkSize.y+1, chunkSize.z+1, blocks);
        UpdateChunk();
    }
    
    [ServerRpc(RequireOwnership = false)]
    public void DestroyBlockServerRpc(Vector3 hit) {
        Debug.Log("DESTROY BLOCK PER RPC");
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].Empty = true;
        DestroyBlockSerializedClientRpc(FlattenBlocks());
    }

    [ClientRpc]
    private void DestroyBlockSerializedClientRpc(Block[] blocks) {
        Debug.Log("DESTROY BLOCK SERIALIZED CLIENT RPC");
        chunkBlocks = ExpandBlocks(chunkSize.x+1, chunkSize.y+1, chunkSize.z+1, blocks);
        UpdateChunk();
    }
    
    // [ClientRpc]
    // private void DestroyBlockClientRpc(Vector3 hit) {
    //     Debug.Log("DESTROY BLOCK PER CLIENT RPC");
    //     chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].Empty = true;
    //     
    //     var flattened = FlattenBlocks();
    //     var unflattened = ExpandBlocks(chunkBlocks.GetLength(0), chunkBlocks.GetLength(1), chunkBlocks.GetLength(2), flattened);
    //     Debug.Log(unflattened);
    //     chunkBlocks = unflattened;
    //     UpdateChunk();
    //
    // }

    public void BuildBlockServer(Vector3 hit) {
        if (!IsServer)
            Debug.Log("SERVER BUILD BLOCK NOT CALLED BY OWNER - THIS SHOULD NEVER HAPPEN");
        Debug.Log("NORMAL BUILD BLOCK: " + hit.x + " " + hit.y + " " + hit.z);
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].Empty = false;
        Debug.Log("Calling the serialized method");
        BuildBlockSerializedClientRpc(FlattenBlocks());
        // Chunk Size debug
        // int sendSize = 2000;
        // Block[] sendVec = new Block[sendSize];
        // var vectorHurensohn = FlattenBlocks();
        // for (int i = 0; i < sendSize; i++) {
        //     sendVec[i] = vectorHurensohn[i];
        // }
        // BuildBlockSerializedClientRpc(sendVec);
    }

    [ClientRpc]
    private void BuildBlockSerializedClientRpc(Block[] blocks) {
        Debug.Log("BUILD BLOCK SERIALIZED CLIENT RPC");
        Debug.Log(blocks);
        chunkBlocks = ExpandBlocks(chunkSize.x+1, chunkSize.y+1, chunkSize.z+1, blocks);
        UpdateChunk();
    }

    // [ClientRpc]
    // private void BuildBlockClientRpc(Vector3 hit) {
    //     Debug.Log("Client RPC BUILD BLOCK: " + hit.x + " " + hit.y + " " + hit.z);
    //     chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].Empty = false;
    //     UpdateChunk();
    // }
    
    public void DamageBlock(Vector3 hit, byte damage) {
        Debug.Log(hit.x + " " + hit.y + " " + hit.z);
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].DamageBlock(damage);
        Debug.Log("Damaging");
        UpdateChunk();
    }

    public Block[] FlattenBlocks() {
        return chunkBlocks.Cast<Block>().ToArray();
    }
    
    // https://stackoverflow.com/questions/62381835/how-to-get-a-2d-array-of-strings-across-the-photon-network-in-a-unity-multi-play
    private Block[,,] ExpandBlocks(int elCountX, int elCountY, int elSize, Block[] blocks) {
        Block[,,] expBlocks = new Block[elCountX, elCountY, elSize];
        for (var x = 0; x < elCountX; x++) {
            for (var y = 0; y < elCountY; y++) {
                for (var z = 0; z < elSize; z++) {
                    expBlocks[x, y, z] = blocks[(x * elCountY * elSize) + (y * elSize) + z];
                }    
            }
        }
        return expBlocks;
    }
    
    // [ServerRpc(RequireOwnership = false)]
    // public void BuildBlockServerRpc(Vector3 hit) {
    //     Debug.Log(hit.x + " " + hit.y + " " + hit.z);
    //
    //     chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].empty = false;
    //     Debug.Log("BUILDING");
    //     UpdateChunk();
    // }
    
    // public void DestroyBlock(Vector3 hit) {
    //     Debug.Log(hit.x + " " + hit.y + " " + hit.z);
    //     chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].empty = true;
    //     UpdateChunk();
    // }
    
    public void Serialize(string mapName, int x, int y) {
        BinaryFormatter bf = new BinaryFormatter();
        string savePath = Application.persistentDataPath + Path.DirectorySeparatorChar + mapName + Path.DirectorySeparatorChar + mapName + "_" + x + y + ".chunk";

        Directory.CreateDirectory(Application.persistentDataPath + Path.DirectorySeparatorChar + mapName);

        using var fileStream = File.Create(savePath);
        bf.Serialize(fileStream, chunkBlocks);

        Debug.Log(savePath);
        Debug.Log("Serialized");
    }

    public void Load(string mapName, int x, int y) {
        string loadPath = Application.persistentDataPath + Path.DirectorySeparatorChar + mapName + Path.DirectorySeparatorChar + mapName + "_" + x + y + ".chunk";

        if (File.Exists(loadPath)) {
            BinaryFormatter bf = new BinaryFormatter();
            using var fileStream = File.Open(loadPath, FileMode.Open);
            var updatedBlocks = (Block[,,]) bf.Deserialize(fileStream);

            Debug.Log("Deserialized");
            chunkBlocks = updatedBlocks;
            UpdateChunk();
        }
        else {
            Debug.Log("Error: Chunk.cs: Chunk file was not found");
        }
    }

    private void GenerateChunk() {
        float[,] chunkHeights = Noise.Generate(chunkSize.x + 1, chunkSize.y + 1, seed, intensity);
        chunkBlocks = new Block[chunkSize.x + 1, chunkSize.y + 1, chunkSize.z + 1];

        for (int x = 0; x <= chunkSize.x; x++) {
            for (int z = 0; z <= chunkSize.z; z++) {
                for (int y = 0; y <= chunkSize.y; y++) {
                    chunkBlocks[x, y, z] = new Block(true);

                    if (y <= chunkHeights[x, z]) {
                        chunkBlocks[x, y, z] = new Block(false);
                        chunkBlocks[x, y, z].id = (byte) Random.Range(0, 4);
                        //Debug.Log("Creating Block with id" + chunkBlocks[x, y, z].id);
                    }
                }
            }
        }

        UpdateChunk();
    }

    private void AddVertices(int x, int y, int z) {
        if (CheckSides(new RVector3(x, y, z), BlockFace.Top)) {
            verticesIndex = chunkVertices.Count;


            chunkVertices.Add(new Vector3(x, y + blockSize, z));
            chunkVertices.Add(new Vector3(x, y + blockSize, z + blockSize));
            chunkVertices.Add(new Vector3(x + blockSize, y + blockSize, z + blockSize));
            chunkVertices.Add(new Vector3(x + blockSize, y + blockSize, z));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }

        if (CheckSides(new RVector3(x, y, z), BlockFace.Bottom)) {
            verticesIndex = chunkVertices.Count;

            chunkVertices.Add(new Vector3(x, y, z));
            chunkVertices.Add(new Vector3(x + blockSize, y, z));
            chunkVertices.Add(new Vector3(x + blockSize, y, z + blockSize));
            chunkVertices.Add(new Vector3(x, y, z + blockSize));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }

        if (CheckSides(new RVector3(x, y, z), BlockFace.Right)) {
            verticesIndex = chunkVertices.Count;

            chunkVertices.Add(new Vector3(x + blockSize, y, z));
            chunkVertices.Add(new Vector3(x + blockSize, y + blockSize, z));
            chunkVertices.Add(new Vector3(x + blockSize, y + blockSize, z + blockSize));
            chunkVertices.Add(new Vector3(x + blockSize, y, z + blockSize));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }

        if (CheckSides(new RVector3(x, y, z), BlockFace.Left)) {
            verticesIndex = chunkVertices.Count;

            chunkVertices.Add(new Vector3(x, y, z + blockSize));
            chunkVertices.Add(new Vector3(x, y + blockSize, z + blockSize));
            chunkVertices.Add(new Vector3(x, y + blockSize, z));
            chunkVertices.Add(new Vector3(x, y, z));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }

        if (CheckSides(new RVector3(x, y, z), BlockFace.Far)) {
            verticesIndex = chunkVertices.Count;

            chunkVertices.Add(new Vector3(x, y, z + blockSize));
            chunkVertices.Add(new Vector3(x + blockSize, y, z + blockSize));
            chunkVertices.Add(new Vector3(x + blockSize, y + blockSize, z + blockSize));
            chunkVertices.Add(new Vector3(x, y + blockSize, z + blockSize));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }

        if (CheckSides(new RVector3(x, y, z), BlockFace.Near)) {
            verticesIndex = chunkVertices.Count;

            chunkVertices.Add(new Vector3(x, y, z));
            chunkVertices.Add(new Vector3(x, y + blockSize, z));
            chunkVertices.Add(new Vector3(x + blockSize, y + blockSize, z));
            chunkVertices.Add(new Vector3(x + blockSize, y, z));

            UpdateChunkUV(chunkBlocks[x, y, z].id);
        }
    }

    private void UpdateChunk() {
        //Debug.Log("start updating " + Time.time * 1000);

        chunkVertices = new List<Vector3>();
        chunkUV = new List<Vector2>();
        chunkTriangles = new List<int>();

        chunkMesh.Clear();

        for (var y = 0; y <= chunkSize.y; y++) {
            for (var x = 0; x <= chunkSize.x; x++) {
                for (var z = 0; z <= chunkSize.z; z++) {
                    if (!chunkBlocks[x, y, z].Empty) {
                        AddVertices(x, y, z);
                    }
                }
            }
        }

        //Debug.Log("before finzalize "+  +Time.time * 1000);
        FinalizeChunk();

        DestroyImmediate(gameObject.GetComponent<MeshCollider>());
        MeshCollider mc = gameObject.AddComponent<MeshCollider>();
        mc.material = worldMaterial;

        Debug.Log("updated chunk");
    }

    private bool CheckSides(RVector3 blockPosition, BlockFace blockFace) {
        int x, y, z;
        x = blockPosition.x;
        y = blockPosition.y;
        z = blockPosition.z;
        
        switch (blockFace) {
            case BlockFace.Top: //Checks top face
                if (y + 1 <= chunkSize.y) {
                    //Debug.Log(chunkBlocks[x, y + 1, z].empty);
                    if (!chunkBlocks[x, y + 1, z].Empty) {
                        return false;
                    }
                }
                break;


            case BlockFace.Bottom: //Checks bottom face
                if (y - 1 >= 0) {
                    if (!chunkBlocks[x, y - 1, z].Empty) {
                        return false;
                    }
                }
                break;

            case BlockFace.Right: //Checks right face


                if (x + 1 <= chunkSize.x) {
                    if (!chunkBlocks[x + 1, y, z].Empty) {
                        return false;
                    }
                }
                break;

            case BlockFace.Left: //Checks Left face

                if (x - 1 >= 0) {
                    if (!chunkBlocks[x - 1, y, z].Empty) {
                        return false;
                    }
                }
                break;

            case BlockFace.Far: //Checks Far face

                if (z + 1 <= chunkSize.z) {
                    if (!chunkBlocks[x, y, z + 1].Empty) {
                        return false;
                    }
                }
                break;


            case BlockFace.Near: //Checks Near face
                if (z - 1 >= 0) {
                    if (!chunkBlocks[x, y, z - 1].Empty) {
                        return false;
                    }
                }
                break;
        }

        return true;
    }

    private void UpdateChunkUV(byte blockID) {
        chunkTriangles.Add(verticesIndex);
        chunkTriangles.Add(verticesIndex + 1);
        chunkTriangles.Add(verticesIndex + 2);

        chunkTriangles.Add(verticesIndex + 2);
        chunkTriangles.Add(verticesIndex + 3);
        chunkTriangles.Add(verticesIndex);

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

    private void FinalizeChunk() {
        chunkMesh.vertices = chunkVertices.ToArray();
        chunkMesh.triangles = chunkTriangles.ToArray();
        chunkMesh.uv = chunkUV.ToArray();
        chunkMesh.RecalculateNormals();
    }
    
}