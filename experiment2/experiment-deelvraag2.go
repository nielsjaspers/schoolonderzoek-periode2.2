package experiment2

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Edge vertegenwoordigt een verbinding tussen twee knooppunten met een gewicht (kosten).
type Edge struct {
	To     int
	Weight int
}

// Node vertegenwoordigt een knooppunt in de graaf.
type Node struct {
	ID    int
	Edges []Edge
	X     float64 // X-coördinaat
	Y     float64 // Y-coördinaat
}

// Graph vertegenwoordigt de graafstructuur.
type Graph struct {
	Nodes []Node
}

// PriorityQueueItem vertegenwoordigt een item in de prioriteitswachtrij voor A* en Dijkstra.
type PriorityQueueItem struct {
	NodeID   int
	Priority int
	Index    int // Nodig voor heap.Interface methoden
}

// PriorityQueue implementeert een prioriteitswachtrij gebaseerd op een heap.
type PriorityQueue []*PriorityQueueItem

func (pq PriorityQueue) Len() int {
	return len(pq)
}

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].Priority < pq[j].Priority
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].Index = i
	pq[j].Index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*PriorityQueueItem)
	item.Index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil  // voorkom geheugenlek
	item.Index = -1 // voor de veiligheid
	*pq = old[0 : n-1]
	return item
}

// update wijzigt de prioriteit van een item in de prioriteitswachtrij.
func (pq *PriorityQueue) update(item *PriorityQueueItem, nodeID int, priority int) {
	item.NodeID = nodeID
	item.Priority = priority
	heap.Fix(pq, item.Index)
}

// Heuristieken
type Heuristic func(nodeID, targetID int, graph Graph) int

// Euclidische afstandsheuristiek.
func euclideanHeuristic(nodeID, targetID int, graph Graph) int {
	node := graph.Nodes[nodeID]
	target := graph.Nodes[targetID]
	dx := node.X - target.X
	dy := node.Y - target.Y
	return int(math.Sqrt(dx*dx + dy*dy))
}

// Manhattan afstandsheuristiek.
func manhattanHeuristic(nodeID, targetID int, graph Graph) int {
	node := graph.Nodes[nodeID]
	target := graph.Nodes[targetID]
	dx := math.Abs(node.X - target.X)
	dy := math.Abs(node.Y - target.Y)
	return int(dx + dy)
}

// Nul heuristiek (Dijkstra).
func nullHeuristic(nodeID, targetID int, graph Graph) int {
	return 0
}

// Dijkstra implementatie.
func Dijkstra(graph Graph, startNode, endNode int) (int, []int, time.Duration) {
	startTime := time.Now()

	distances := make([]int, len(graph.Nodes))
	previousNodes := make([]int, len(graph.Nodes))
	for i := range distances {
		distances[i] = math.MaxInt32
		previousNodes[i] = -1
	}
	distances[startNode] = 0

	pq := make(PriorityQueue, 0)
	heap.Init(&pq)
	heap.Push(&pq, &PriorityQueueItem{NodeID: startNode, Priority: 0})

	for pq.Len() > 0 {
		currentItem := heap.Pop(&pq).(*PriorityQueueItem)
		currentNodeID := currentItem.NodeID

		if currentNodeID == endNode {
			break
		}

		for _, edge := range graph.Nodes[currentNodeID].Edges {
			newDist := distances[currentNodeID] + edge.Weight
			if newDist < distances[edge.To] {
				distances[edge.To] = newDist
				previousNodes[edge.To] = currentNodeID

				// Update of voeg toe aan de prioriteitswachtrij
				found := false
				for _, item := range pq {
					if item.NodeID == edge.To {
						pq.update(item, edge.To, newDist)
						found = true
						break
					}
				}
				if !found {
					heap.Push(&pq, &PriorityQueueItem{NodeID: edge.To, Priority: newDist})
				}
			}
		}
	}

	endTime := time.Now()
	duration := endTime.Sub(startTime)

	// Reconstruct path
	path := reconstructPath(previousNodes, endNode)

	return distances[endNode], path, duration
}

// AStar implementatie met een generieke heuristiek.
func AStar(graph Graph, startNode, endNode int, heuristic Heuristic) (int, []int, time.Duration) {
	startTime := time.Now()

	distances := make([]int, len(graph.Nodes))
	previousNodes := make([]int, len(graph.Nodes))
	for i := range distances {
		distances[i] = math.MaxInt32
		previousNodes[i] = -1
	}
	distances[startNode] = 0

	pq := make(PriorityQueue, 0)
	heap.Init(&pq)
	heap.Push(&pq, &PriorityQueueItem{NodeID: startNode, Priority: 0 + heuristic(startNode, endNode, graph)})

	for pq.Len() > 0 {
		currentItem := heap.Pop(&pq).(*PriorityQueueItem)
		currentNodeID := currentItem.NodeID

		if currentNodeID == endNode {
			break
		}

		for _, edge := range graph.Nodes[currentNodeID].Edges {
			newDist := distances[currentNodeID] + edge.Weight
			if newDist < distances[edge.To] {
				distances[edge.To] = newDist
				previousNodes[edge.To] = currentNodeID
				priority := newDist + heuristic(edge.To, endNode, graph)

				// Update of voeg toe aan de prioriteitswachtrij
				found := false
				for _, item := range pq {
					if item.NodeID == edge.To {
						pq.update(item, edge.To, priority)
						found = true
						break
					}
				}
				if !found {
					heap.Push(&pq, &PriorityQueueItem{NodeID: edge.To, Priority: priority})
				}
			}
		}
	}

	endTime := time.Now()
	duration := endTime.Sub(startTime)

	// Reconstruct path
	path := reconstructPath(previousNodes, endNode)

	return distances[endNode], path, duration
}

// reconstructPath reconstrueert het pad van de vorige knooppunten.
func reconstructPath(previousNodes []int, endNode int) []int {
	path := []int{}
	for current := endNode; current != -1; current = previousNodes[current] {
		path = append([]int{current}, path...)
	}
	return path
}

// createGraph genereert een voorbeeldgraaf.
func createGraph(numNodes int, edgeDensity float64, trafficCondition string) Graph {
	graph := Graph{Nodes: make([]Node, numNodes)}

	for i := 0; i < numNodes; i++ {
		x := rand.Float64() * 100 // Willekeurige X-coördinaat
		y := rand.Float64() * 100 // Willekeurige Y-coördinaat
		graph.Nodes[i] = Node{ID: i, Edges: []Edge{}, X: x, Y: y}
		for j := 0; j < numNodes; j++ {
			if i != j && rand.Float64() < edgeDensity {
				weight := generateWeight(trafficCondition)
				graph.Nodes[i].Edges = append(graph.Nodes[i].Edges, Edge{To: j, Weight: weight})
			}
		}
	}

	return graph
}

// generateWeight genereert een gewicht op basis van de verkeersomstandigheden.
func generateWeight(trafficCondition string) int {
	switch trafficCondition {
	case "Laag verkeer":
		return rand.Intn(5) + 1 // 1-5
	case "Hoog verkeer":
		return rand.Intn(50) + 20 // 20-70
	case "Gemengd verkeer":
		if rand.Float64() < 0.5 {
			return rand.Intn(5) + 1 // 1-5
		}
		return rand.Intn(50) + 20 // 20-70
	default:
		return rand.Intn(20) + 1 // 1-20
	}
}

// simulateIncident simuleert een incident door het gewicht van een specifieke weg te verhogen.
func simulateIncident(graph *Graph, fromNode, toNode, increasedWeight int) {
	for i := range graph.Nodes[fromNode].Edges {
		if graph.Nodes[fromNode].Edges[i].To == toNode {
			graph.Nodes[fromNode].Edges[i].Weight = increasedWeight
			return
		}
	}
	// Als de edge niet bestaat, voeg deze toe.
	graph.Nodes[fromNode].Edges = append(graph.Nodes[fromNode].Edges, Edge{To: toNode, Weight: increasedWeight})
}

// adjustWeights past de gewichten van de verbindingen aan op basis van een factor.
func adjustWeights(graph *Graph, adjustmentFactor float64) {
	for i := range graph.Nodes {
		for j := range graph.Nodes[i].Edges {
			graph.Nodes[i].Edges[j].Weight = int(float64(graph.Nodes[i].Edges[j].Weight) * adjustmentFactor)
		}
	}
}

func Experiment2() {
	rand.Seed(time.Now().UnixNano())

	numNodes := 10000
	edgeDensity := 0.2
	startNode := 0
	endNode := 99

	trafficConditions := []string{"Laag verkeer", "Hoog verkeer", "Gemengd verkeer"}
	heuristics := map[string]Heuristic{
		"Null":      nullHeuristic,
		"Euclidean": euclideanHeuristic,
		"Manhattan": manhattanHeuristic,
	}
	weightAdjustments := []float64{0.5, 1.0, 1.5}

	for _, condition := range trafficConditions {
		fmt.Printf("Verkeersomstandigheden: %s\n", condition)
		graph := createGraph(numNodes, edgeDensity, condition)

		// Incident simulatie
		if condition == "Gemengd verkeer" {
			simulateIncident(&graph, 20, 30, 100) // Simulatie van een incident
			fmt.Println("Incident gesimuleerd tussen knooppunt 20 en 30")
		}

		for heuristicName, heuristic := range heuristics {
			fmt.Printf("  Heuristiek: %s\n", heuristicName)

			// Dijkstra (geen heuristiek)
			if heuristicName == "Null" {
				distanceDijkstra, pathDijkstra, durationDijkstra := Dijkstra(graph, startNode, endNode)
				fmt.Printf("    Dijkstra: Afstand = %d, Pad = %v, Tijd = %s\n", distanceDijkstra, pathDijkstra, durationDijkstra)
			}

			// A* met de huidige heuristiek
			distanceAStar, pathAStar, durationAStar := AStar(graph, startNode, endNode, heuristic)
			fmt.Printf("    A*: Afstand = %d, Pad = %v, Tijd = %s\n", distanceAStar, pathAStar, durationAStar)

			// Gewichtsaanpassingen
			for _, adjustmentFactor := range weightAdjustments {
				fmt.Printf("    Gewichtsaanpassing: %.1f\n", adjustmentFactor)
				graphCopy := graph // Maak een kopie van de graaf
				adjustWeights(&graphCopy, adjustmentFactor)

				// Dijkstra (geen heuristiek)
				if heuristicName == "Null" {
					distanceDijkstra, pathDijkstra, durationDijkstra := Dijkstra(graphCopy, startNode, endNode)
					fmt.Printf("      Dijkstra: Afstand = %d, Pad = %v, Tijd = %s\n", distanceDijkstra, pathDijkstra, durationDijkstra)
				}

				// A* met de huidige heuristiek
				distanceAStar, pathAStar, durationAStar := AStar(graphCopy, startNode, endNode, heuristic)
				fmt.Printf("      A*: Afstand = %d, Pad = %v, Tijd = %s\n", distanceAStar, pathAStar, durationAStar)
			}
		}
		fmt.Println()
	}
}

