using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Raycast : MonoBehaviour
{
    private Camera _camera;
    private GameObject _world;

    // Start is called before the first frame update
    void Start()
    {
        _world = GameObject.Find("world");
        _camera = GetComponentInChildren<Camera>();
    }

    // Update is called once per frame
    void Update()
    {
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
                    GameObject chunk = hit.transform.gameObject;
                    Vector3 localCoordinate = hit.point - chunk.transform.position;
                    chunk.GetComponent<Chunk>().DestroyBlock(localCoordinate);
                    Destroy(chunk.GetComponent<MeshCollider>());
                    MeshCollider mc = chunk.AddComponent<MeshCollider>();
                    mc.material = _world.GetComponent<World>().worldMaterial;
                    
                    StartCoroutine(HitIndicator(hit.point));
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
