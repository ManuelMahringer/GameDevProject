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
    public Block[,,] chunkBlocks;

    private World _world;
    private Vector3 pos;
    private Mesh chunkMesh;
    private List<Vector3> chunkVertices = new List<Vector3>();
    private List<Vector2> chunkUV = new List<Vector2>();
    private List<int> chunkTriangles = new List<int>();
    private int verticesIndex;
    private Texture textureAtlas;
    private Vector2 atlasSize;

    void Awake() {
        _world = GameObject.Find("World").GetComponent<World>();
        textureAtlas = transform.GetComponent<MeshRenderer>().material.mainTexture;
        
        float worldSize = _world.size * _world.chunkSize;
        int posX = (int) ((transform.position.x + worldSize / 2) / _world.chunkSize);
        int posY = (int) ((transform.position.y + _world.height * _world.chunkSize / 2) / _world.chunkSize);
        int posZ = (int) ((transform.position.z + worldSize / 2) / _world.chunkSize);
        pos = new Vector3(posX, posY, posZ);
            
        atlasSize = new Vector2(textureAtlas.width / textureBlockSize.x, textureAtlas.height / textureBlockSize.y);
        chunkMesh = GetComponent<MeshFilter>().mesh;
        chunkMesh.MarkDynamic();
        if (IsOwnedByServer) {
            if (_world.selectedMap == "Generate") {
                // dummy option to still be able to generate the random map TODO: remove
                GenerateChunk(posY);
                Debug.Log("GENERATING MAP");
            }
            else {
                // he host was für eine map junge
                Debug.Log("Selected World in Chunk " + _world.selectedMap);
                Load(_world.selectedMap, (int) pos.x, (int) pos.y, (int) pos.z);
            }
        }
    }
    
    
    [ServerRpc(RequireOwnership = false)]
    public void DestroyBlockServerRpc(Vector3 hit) {
        Debug.Log("DESTROY BLOCK PER RPC");
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].Empty = true;
        UpdateChunkClientRpc(FlattenBlocks());
    }

    public void BuildBlockServer(Vector3 hit, BlockType blockType) {
        Debug.Log("NORMAL BUILD BLOCK: " + hit.x + " " + hit.y + " " + hit.z);
        Block block = chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)];
        block.id = (byte) blockType;
        block.Empty = false;
        Debug.Log("Calling the serialized method");
        UpdateChunkClientRpc(FlattenBlocks());
        // Chunk Size debug
        // int sendSize = 2000;
        // Block[] sendVec = new Block[sendSize];
        // var flat = FlattenBlocks();
        // for (int i = 0; i < sendSize; i++) {
        //     sendVec[i] = flat[i];
        // }
        // BuildBlockSerializedClientRpc(sendVec);
    }
    
    [ServerRpc(RequireOwnership = false)]
    public void DamageBlockServerRpc(Vector3 hit, sbyte damage) {
        Debug.Log(hit.x + " " + hit.y + " " + hit.z);
        chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].DamageBlock(damage);
        Debug.Log("Damaging");
        UpdateChunkClientRpc(FlattenBlocks());
    }
    
    [ClientRpc]
    private void UpdateChunkClientRpc(Block[] blocks) {
        chunkBlocks = ExpandBlocks(chunkSize.x + 1, chunkSize.y + 1, chunkSize.z + 1, blocks);
        UpdateChunk();
    }

    // Flatten 3D block array to 1D to be able to send it over network
    private Block[] FlattenBlocks() {
        return chunkBlocks.Cast<Block>().ToArray();
    }
    
    // Expand the flattened 1D block array back to 3D
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
    // [ClientRpc]
    // private void DestroyBlockClientRpc(Vector3 hit) {
    //     Debug.Log("DESTROY BLOCK PER CLIENT RPC");
    //     chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].Empty = true;
    //     
    //     UpdateChunk();
    //
    // }
    // [ClientRpc]
    // private void BuildBlockClientRpc(Vector3 hit) {
    //     Debug.Log("Client RPC BUILD BLOCK: " + hit.x + " " + hit.y + " " + hit.z);
    //     chunkBlocks[Mathf.FloorToInt(hit.x), Mathf.FloorToInt(hit.y), Mathf.FloorToInt(hit.z)].Empty = false;
    //     UpdateChunk();
    // }

    public void Serialize(string mapName, int x, int y, int z) {
        BinaryFormatter bf = new BinaryFormatter();
        string basePath = Directory.GetCurrentDirectory() + Path.DirectorySeparatorChar + "Maps";
        string savePath = basePath + Path.DirectorySeparatorChar + mapName + Path.DirectorySeparatorChar + mapName + "_" + x + "_"+ y + "_" + z +".chunk";

        Directory.CreateDirectory(basePath + Path.DirectorySeparatorChar + mapName);

        using var fileStream = File.Create(savePath);
        bf.Serialize(fileStream, chunkBlocks);

        Debug.Log(savePath);
        //Debug.Log("Serialized");
    }

    public void Load(string mapName, int x, int y, int z) {
        string basePath = Directory.GetCurrentDirectory() + Path.DirectorySeparatorChar + "Maps";
        string loadPath = basePath + Path.DirectorySeparatorChar + mapName + Path.DirectorySeparatorChar + mapName + "_" + x + "_"+ y + "_" + z +".chunk";

        if (File.Exists(loadPath)) {
            BinaryFormatter bf = new BinaryFormatter();
            using var fileStream = File.Open(loadPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            var updatedBlocks = (Block[,,]) bf.Deserialize(fileStream);

            //Debug.Log("Deserialized");
            chunkBlocks = updatedBlocks;
            UpdateChunk();
        }
        else {
            Debug.Log("Error: Chunk.cs: Chunk file was not found " + loadPath);
        }
    }

    private void GenerateChunk(int height) {
        Debug.Log("calling generate with height " + height);
        float[,] chunkHeights = Noise.Generate(chunkSize.x + 1, chunkSize.y + 1, seed, intensity);
        chunkBlocks = new Block[chunkSize.x + 1, chunkSize.y + 1, chunkSize.z + 1];

        for (int x = 0; x <= chunkSize.x; x++) {
            for (int z = 0; z <= chunkSize.z; z++) {
                for (int y = 0; y <= chunkSize.y; y++) {
                    chunkBlocks[x, y, z] = new Block(true, 0);
                    if (height <= 2) {
                        chunkBlocks[x, y, z] = new Block(false, (byte) Random.Range(0, 4));
                        //chunkBlocks[x, y, z].id = (byte) Random.Range(0, 4);
                        //Debug.Log("Creating Block with id" + chunkBlocks[x, y, z].id);
                    }
                    else {
                        if(y <= chunkHeights[x, z])
                            chunkBlocks[x, y, z] = new Block(false, (byte) Random.Range(0, 4));
                        
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

        //Debug.Log("updated chunk");
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