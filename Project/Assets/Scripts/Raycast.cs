using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Raycast : MonoBehaviour
{
    private Camera _camera;
    private GameObject _world;
   // private LineRenderer _laserLine;

    // Start is called before the first frame update
    void Start()
    {
        _world = GameObject.Find("world");
        _camera = GetComponentInChildren<Camera>();
        //_laserLine = GetComponent<LineRenderer>();,
            
    }

    // Update is called once per frame
    void Update()
    {
        // Remove block with left click
        if(Input.GetMouseButtonDown(0))
        {
            // Pos middle of screen
            Vector3 midPoint = new Vector3(_camera.pixelWidth / 2, _camera.pixelHeight / 2);
            //ray passing mittlde of screen
            Ray ray = _camera.ScreenPointToRay(midPoint);
            RaycastHit hit;
            Debug.Log("SHOOOOOOOOOOOOOOT");
              
            if (Physics.Raycast(ray, out hit))
            {
                if(hit.transform.gameObject.GetComponent<Chunk>() != null) // Check if its a chunk
                {
                    Debug.Log("Hit: " + hit.point);
                    Debug.Log("Ray " + ray);
                    GameObject chunk = hit.transform.gameObject;
                    // try to shift hit.point a little bit to counter the index out of bounds error

                    Vector3 localCoordinate = hit.point + (ray.direction/10000.0f) - chunk.transform.position;
                    chunk.GetComponent<Chunk>().DestroyBlock(localCoordinate);
                    Destroy(chunk.GetComponent<MeshCollider>());
                    MeshCollider mc = chunk.AddComponent<MeshCollider>();
                    mc.material = _world.GetComponent<World>().worldMaterial;
                    
                    //StartCoroutine(HitIndicator(hit.point));
                    //_world.GetComponent<World>().DestroyBlock(hit.point);
                }
            }
        }
        // Build block with right click
        // Take ray at fixed range - find chunk based on coords and invoke buildBlock with coords
        if (Input.GetMouseButtonDown(1))
        {
            // Pos middle of screen
            Vector3 midPoint = new Vector3(_camera.pixelWidth / 2, _camera.pixelHeight / 2);
            //ray passing mittlde of screen
            Ray ray = _camera.ScreenPointToRay(midPoint);
            RaycastHit hit;
            Debug.Log("SHOOOOOOOOOOOOOOT");
            if (Physics.Raycast(ray, out hit))
            {

                if (hit.transform.gameObject.GetComponent<Chunk>() != null) // Check if its a chunk
                {
                    Debug.Log("Hit: " + hit.point);
                    Debug.Log("Ray " + ray);
                    //GameObject chunk = hit.transform.gameObject;
                    // try to shift hit.point a little bit to counter the index out of bounds error

                    
                    GameObject chunk = GameObject.Find("world").GetComponent<World>().FindChunk(hit.point - (ray.direction / 10000.0f));
                    Vector3 localCoordinate = hit.point - (ray.direction / 10000.0f) - chunk.transform.position;
                    Debug.Log("localCoordinate" + localCoordinate);
                    chunk.GetComponent<Chunk>().BuildBlock(localCoordinate);
                    Destroy(chunk.GetComponent<MeshCollider>());
                    MeshCollider mc = chunk.AddComponent<MeshCollider>();
                    mc.material = _world.GetComponent<World>().worldMaterial;

                    //StartCoroutine(HitIndicator(hit.point));
                    //_world.GetComponent<World>().DestroyBlock(hit.point);
                }
            }
        }

    }

    private IEnumerator HitIndicator(Vector3 hitLocation)
    {
        GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.transform.position = hitLocation;

        yield return new WaitForSeconds(1.0f);

        Destroy(sphere);
    }
}
