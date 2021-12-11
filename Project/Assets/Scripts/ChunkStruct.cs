using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;
using Unity.Netcode;

public struct ChunkStruct : INetworkSerializable
{
    List<Vector3> chunkVerticies;
    List<Vector2> chunkUV;
    List<int> chunkTriangles;

    // INetworkSerializable
    public void NetworkSerialize<T>(BufferSerializer<T> serializer) where T : IReaderWriter
    {
        BinaryFormatter formatter = new BinaryFormatter();
        System.IO.MemoryStream ms = new System.IO.MemoryStream();
        formatter.Serialize(ms,chunkVerticies);
        for (int i = 0; i < ms.ToArray().Length;  i ++)
        {
            serializer.SerializeValue(ref ms.ToArray()[0]);
        }
        formatter.Serialize(ms,chunkUV);
        for (int i = 0; i < ms.ToArray().Length;  i ++)
        {
            serializer.SerializeValue(ref ms.ToArray()[0]);
        }
        
        
        
    }
    // ~INetworkSerializable
}