using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class World : MonoBehaviour
{
    // Reference to the Chunk Prefab. Drag a Prefab into this field in the Inspector.
    [SerializeField] 
    private GameObject myPrefab;
    public PhysicMaterial worldMaterial;
    [SerializeField]
    private int size;
    [SerializeField]
    private float chunk_size;



    private GameObject[,] chunks;

    void Awake()
    {
        worldMaterial = new PhysicMaterial
        {
            staticFriction = 0f,
            dynamicFriction = 0f,
            bounciness = 0f,
            frictionCombine = PhysicMaterialCombine.Minimum,
            bounceCombine = PhysicMaterialCombine.Minimum
        };
    }

    public GameObject FindChunk(Vector3 worldCoordinate)
    {
        Debug.Log("found chunk " + Mathf.FloorToInt(worldCoordinate.x / chunk_size) + " " +  Mathf.FloorToInt(worldCoordinate.z / chunk_size));
        return chunks[Mathf.Abs(Mathf.FloorToInt(worldCoordinate.x / chunk_size)), Mathf.Abs(Mathf.FloorToInt(worldCoordinate.z / chunk_size))];
    } 

    private void addMeshCollider(int x, int z) {
        MeshCollider mc = chunks[x, z].AddComponent<MeshCollider>();
        mc.material = worldMaterial;
    }


    void Start()
    {
        chunks = new GameObject[size,size]; 
        // Instantiate chunks
        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < size; y++)
            {
                Debug.Log("instantiate now");

                chunks[x, y] = Instantiate(myPrefab, new Vector3(chunk_size * x,1, 17 * y), Quaternion.identity); //  This quaternion corresponds to "no rotation" - the object is perfectly aligned with the world or parent axes.
                addMeshCollider(x, y);
            }
        }
    }
} 